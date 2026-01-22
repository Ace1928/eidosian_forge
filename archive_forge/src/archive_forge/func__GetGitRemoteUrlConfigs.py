import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def _GetGitRemoteUrlConfigs(source_directory):
    """Calls git to output every configured remote URL.

  Args:
    source_directory: The path to directory containing the source code.
  Returns:
    The raw output of the command, or None if the command failed.
  """
    return _CallGit(source_directory, 'config', '--get-regexp', _REMOTE_URL_PATTERN)