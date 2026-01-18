import atexit
import json
import os
import socket
import subprocess
import sys
import threading
import warnings
from copy import copy
from contextlib import contextmanager
from pathlib import Path
from shutil import which
import tenacity
import plotly
from plotly.files import PLOTLY_DIR, ensure_writable_plotly_dir
from plotly.io._utils import validate_coerce_fig_to_dict
from plotly.optional_imports import get_module
def validate_executable():
    """
    Attempt to find and validate the orca executable specified by the
    `plotly.io.orca.config.executable` property.

    If the `plotly.io.orca.status.state` property is 'validated' or 'running'
    then this function does nothing.

    How it works:
      - First, it searches the system PATH for an executable that matches the
      name or path specified in the `plotly.io.orca.config.executable`
      property.
      - Then it runs the executable with the `--help` flag to make sure
      it's the plotly orca executable
      - Then it runs the executable with the `--version` flag to check the
      orca version.

    If all of these steps are successful then the `status.state` property
    is set to 'validated' and the `status.executable` and `status.version`
    properties are populated

    Returns
    -------
    None
    """
    if status.state != 'unvalidated':
        return
    install_location_instructions = "If you haven't installed orca yet, you can do so using conda as follows:\n\n    $ conda install -c plotly plotly-orca\n\nAlternatively, see other installation methods in the orca project README at\nhttps://github.com/plotly/orca\n\nAfter installation is complete, no further configuration should be needed.\n\nIf you have installed orca, then for some reason plotly.py was unable to\nlocate it. In this case, set the `plotly.io.orca.config.executable`\nproperty to the full path of your orca executable. For example:\n\n    >>> plotly.io.orca.config.executable = '/path/to/orca'\n\nAfter updating this executable property, try the export operation again.\nIf it is successful then you may want to save this configuration so that it\nwill be applied automatically in future sessions. You can do this as follows:\n\n    >>> plotly.io.orca.config.save()\n\nIf you're still having trouble, feel free to ask for help on the forums at\nhttps://community.plot.ly/c/api/python\n"
    executable = which(config.executable)
    path = os.environ.get('PATH', os.defpath)
    formatted_path = path.replace(os.pathsep, '\n    ')
    if executable is None:
        raise ValueError("\nThe orca executable is required to export figures as static images,\nbut it could not be found on the system path.\n\nSearched for executable '{executable}' on the following path:\n    {formatted_path}\n\n{instructions}".format(executable=config.executable, formatted_path=formatted_path, instructions=install_location_instructions))
    xvfb_args = ['--auto-servernum', '--server-args', '-screen 0 640x480x24 +extension RANDR +extension GLX', executable]
    if config.use_xvfb == True:
        xvfb_run_executable = which('xvfb-run')
        if not xvfb_run_executable:
            raise ValueError("\nThe plotly.io.orca.config.use_xvfb property is set to True, but the\nxvfb-run executable could not be found on the system path.\n\nSearched for the executable 'xvfb-run' on the following path:\n    {formatted_path}".format(formatted_path=formatted_path))
        executable_list = [xvfb_run_executable] + xvfb_args
    elif config.use_xvfb == 'auto' and sys.platform.startswith('linux') and (not os.environ.get('DISPLAY')) and which('xvfb-run'):
        xvfb_run_executable = which('xvfb-run')
        executable_list = [xvfb_run_executable] + xvfb_args
    else:
        executable_list = [executable]
    invalid_executable_msg = "\nThe orca executable is required in order to export figures as static images,\nbut the executable that was found at '{executable}'\ndoes not seem to be a valid plotly orca executable. Please refer to the end of\nthis message for details on what went wrong.\n\n{instructions}".format(executable=executable, instructions=install_location_instructions)
    with orca_env():
        p = subprocess.Popen(executable_list + ['--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        help_result, help_error = p.communicate()
    if p.returncode != 0:
        err_msg = invalid_executable_msg + '\nHere is the error that was returned by the command\n    $ {executable} --help\n\n[Return code: {returncode}]\n{err_msg}\n'.format(executable=' '.join(executable_list), err_msg=help_error.decode('utf-8'), returncode=p.returncode)
        if sys.platform.startswith('linux') and (not os.environ.get('DISPLAY')):
            err_msg += 'Note: When used on Linux, orca requires an X11 display server, but none was\ndetected. Please install Xvfb and configure plotly.py to run orca using Xvfb\nas follows:\n\n    >>> import plotly.io as pio\n    >>> pio.orca.config.use_xvfb = True\n\nYou can save this configuration for use in future sessions as follows:\n\n    >>> pio.orca.config.save()\n\nSee https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml\nfor more info on Xvfb\n'
        raise ValueError(err_msg)
    if not help_result:
        raise ValueError(invalid_executable_msg + '\nThe error encountered is that no output was returned by the command\n    $ {executable} --help\n'.format(executable=' '.join(executable_list)))
    if "Plotly's image-exporting utilities" not in help_result.decode('utf-8'):
        raise ValueError(invalid_executable_msg + '\nThe error encountered is that unexpected output was returned by the command\n    $ {executable} --help\n\n{help_result}\n'.format(executable=' '.join(executable_list), help_result=help_result))
    with orca_env():
        p = subprocess.Popen(executable_list + ['--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        version_result, version_error = p.communicate()
    if p.returncode != 0:
        raise ValueError(invalid_executable_msg + '\nAn error occurred while trying to get the version of the orca executable.\nHere is the command that plotly.py ran to request the version\n    $ {executable} --version\n\nThis command returned the following error:\n\n[Return code: {returncode}]\n{err_msg}\n        '.format(executable=' '.join(executable_list), err_msg=version_error.decode('utf-8'), returncode=p.returncode))
    if not version_result:
        raise ValueError(invalid_executable_msg + '\nThe error encountered is that no version was reported by the orca executable.\nHere is the command that plotly.py ran to request the version:\n\n    $ {executable} --version\n'.format(executable=' '.join(executable_list)))
    else:
        version_result = version_result.decode()
    status._props['executable_list'] = executable_list
    status._props['version'] = version_result.strip()
    status._props['state'] = 'validated'