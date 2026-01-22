from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import hashlib
import itertools
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap
import certifi
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.updater import installers
from googlecloudsdk.core.updater import local_state
from googlecloudsdk.core.updater import release_notes
from googlecloudsdk.core.updater import snapshots
from googlecloudsdk.core.updater import update_check
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
from six.moves import map  # pylint: disable=redefined-builtin
def RestartCommand(command=None, args=None, python=None, block=True):
    """Calls command again with the same arguments as this invocation and exit.

  Args:
    command: str, the command to run (full path to Python file). If not
      specified, defaults to current `gcloud` installation.
    args: list of str or None. If given, use these arguments to the command
      instead of the args for this process.
    python: str or None, the path to the Python interpreter to use for the new
      command invocation (if None, uses the current Python interpreter)
    block: bool, whether to wait for the restarted command invocation to
      terminate before continuing.
  """
    command = command or config.GcloudPath()
    command_args = args or argv_utils.GetDecodedArgv()[1:]
    args = execution_utils.ArgsForPythonTool(command, *command_args, python=python)
    args = [encoding.Encode(a) for a in args]
    short_command = os.path.basename(command)
    if short_command == 'gcloud.py':
        short_command = 'gcloud'
    log_args = ' '.join([console_attr.SafeText(a) for a in command_args])
    log.status.Print('Restarting command:\n  $ {command} {args}\n'.format(command=short_command, args=log_args))
    log.debug('Restarting command: %s %s', command, args)
    log.out.flush()
    log.err.flush()
    if block:
        execution_utils.Exec(args)
    else:
        current_platform = platforms.Platform.Current()
        popen_args = {}
        if console_io.CanPrompt():
            popen_args = current_platform.AsyncPopenArgs()
            if current_platform.operating_system is platforms.OperatingSystem.WINDOWS:

                def Quote(s):
                    return '"' + encoding.Decode(s) + '"'
                args = 'cmd.exe /c "{0} & pause"'.format(' '.join(map(Quote, args)))
        subprocess.Popen(args, shell=True, **popen_args)