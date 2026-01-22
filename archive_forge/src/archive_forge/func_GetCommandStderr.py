import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def GetCommandStderr(command, env=None, close_fds=True):
    """Runs the given shell command and returns a tuple.

  Args:
    command: List or string representing the command to run.
    env: Dictionary of environment variable settings.
    close_fds: Whether or not to close all open fd's in the child after forking.

  Returns:
    Tuple of (exit status, text printed to stdout and stderr by the command).
  """
    if env is None:
        env = {}
    if os.environ.get('PYTHON_RUNFILES') and (not env.get('PYTHON_RUNFILES')):
        env['PYTHON_RUNFILES'] = os.environ['PYTHON_RUNFILES']
    use_shell = isinstance(command, types.StringTypes)
    process = subprocess.Popen(command, close_fds=close_fds, env=env, shell=use_shell, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    output = process.communicate()[0]
    exit_status = process.wait()
    return (exit_status, output)