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