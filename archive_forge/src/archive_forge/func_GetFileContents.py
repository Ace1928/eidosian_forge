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
def GetFileContents(file):
    """Returns the file contents and whether or not the file contains binary data.

  Args:
    file: A file path.

  Returns:
    A tuple of the file contents and whether or not the file contains binary
    contents.
  """
    try:
        contents = file_utils.ReadFileContents(file)
        is_binary = False
    except UnicodeError:
        contents = file_utils.ReadBinaryFileContents(file)
        is_binary = True
    return (contents, is_binary)