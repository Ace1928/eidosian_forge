import json
import logging
import os
import re
import subprocess
from googlecloudsdk.third_party.appengine._internal import six_subset
def HasPendingChanges(source_directory):
    """Checks if the git repo in a directory has any pending changes.

  Args:
    source_directory: The path to directory containing the source code.
  Returns:
    True if there are any uncommitted or untracked changes in the local repo
    for the given directory.
  """
    status = _CallGit(source_directory, 'status')
    return re.search(_GIT_PENDING_CHANGE_PATTERN, status, flags=re.MULTILINE)