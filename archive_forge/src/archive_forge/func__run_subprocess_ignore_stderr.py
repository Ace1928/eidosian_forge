import os
import subprocess
import six
from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
def _run_subprocess_ignore_stderr(command):
    """ Return subprocess.check_output with the given command and ignores stderr."""
    with open(os.devnull, 'w') as devnull:
        output = subprocess.check_output(command, stderr=devnull)
    return output