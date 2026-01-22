import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _CallGit(cwd, *args):
    """Calls git with the given args, in the given working directory.

  Args:
    cwd: The working directory for the command.
    *args: Any arguments for the git command.
  Returns:
    The raw output of the command, or None if the command failed.
  """
    try:
        output = subprocess.check_output(['git'] + list(args), cwd=cwd)
        if six_subset.PY3:
            output = output.decode('utf-8')
        return output
    except (OSError, subprocess.CalledProcessError) as e:
        logging.debug('Could not call git with args %s: %s', args, e)
        return None