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
def GetDirFilesRecursive(directory):
    """Generates the set of all files in directory and its children recursively.

  Args:
    directory: The directory path name.

  Returns:
    A set of all files in directory and its children recursively, relative to
    the directory.
  """
    dirfiles = set()
    for dirpath, _, files in os.walk(six.text_type(directory)):
        for name in files:
            file = os.path.join(dirpath, name)
            relative_file = os.path.relpath(file, directory)
            dirfiles.add(relative_file)
    return dirfiles