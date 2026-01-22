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