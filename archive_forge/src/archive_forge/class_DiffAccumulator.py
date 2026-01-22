from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import os
import shutil
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import parallel
from googlecloudsdk.core.util import text
import six
class DiffAccumulator(object):
    """A module for accumulating DirDiff() differences."""

    def __init__(self):
        self._changes = 0

    def Ignore(self, relative_file):
        """Checks if relative_file should be ignored by DirDiff().

    Args:
      relative_file: A relative file path name to be checked.

    Returns:
      True if path is to be ignored in the directory differences.
    """
        return False

    def AddChange(self, op, relative_file, old_contents=None, new_contents=None):
        """Called for each file difference.

    AddChange() can construct the {'add', 'delete', 'edit'} file operations that
    convert old_dir to match new_dir. Directory differences are ignored.

    This base implementation counts the number of changes.

    Args:
      op: The change operation string;
        'add'; relative_file is not in old_dir.
        'delete'; relative_file is not in new_dir.
        'edit'; relative_file is different in new_dir.
      relative_file: The old_dir and new_dir relative path name of a file that
        changed.
      old_contents: The old file contents.
      new_contents: The new file contents.

    Returns:
      A prune value. If non-zero then DirDiff() returns immediately with that
      value.
    """
        self._changes += 1
        return None

    def GetChanges(self):
        """Returns the accumulated changes."""
        return self._changes

    def Validate(self, relative_file, contents):
        """Called for each file for content validation.

    Args:
      relative_file: The old_dir and new_dir relative path name of an existing
        file.
      contents: The file contents string.
    """
        pass