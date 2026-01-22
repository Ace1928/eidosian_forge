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
class HelpAccumulator(DiffAccumulator):
    """Accumulates help document directory differences.

  Attributes:
    _changes: The list of DirDiff() (op, path) difference tuples.
    _restrict: The set of file path prefixes that the accumulator should be
      restricted to.
  """

    def __init__(self, restrict=None):
        super(HelpAccumulator, self).__init__()
        self._changes = []
        self._restrict = {os.sep.join(r.split('.')[1:]) for r in restrict} if restrict else {}

    def Ignore(self, relative_file):
        """Checks if relative_file should be ignored by DirDiff().

    Args:
      relative_file: A relative file path name to be checked.

    Returns:
      True if path is to be ignored in the directory differences.
    """
        if IsOwnersFile(relative_file):
            return True
        if not self._restrict:
            return False
        for item in self._restrict:
            if relative_file == item or relative_file.startswith(item + os.sep):
                return False
        return True

    def AddChange(self, op, relative_file, old_contents=None, new_contents=None):
        """Adds an DirDiff() difference tuple to the list of changes.

    Args:
      op: The difference operation, one of {'add', 'delete', 'edit'}.
      relative_file: The relative path of a file that has changed.
      old_contents: The old file contents.
      new_contents: The new file contents.

    Returns:
      None which signals DirDiff() to continue.
    """
        self._changes.append((op, relative_file))
        return None