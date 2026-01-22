from the command line arguments and returns a list of URLs to be given to the
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import io
import os
import sys
import textwrap
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.ml_engine import uploads
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
import six
from six.moves import map
from setuptools import setup, find_packages
@contextlib.contextmanager
def _TempDirOrBackup(default_dir):
    """Yields a temporary directory or a backup temporary directory.

  Prefers creating a temporary directory (which will be cleaned up when the
  context manager is closed), but falls back to default_dir. There are systems
  where users can't write to temp, but we still need to copy.

  Args:
    default_dir: str, the backup temporary directory.

  Yields:
    str, the temporary directory.
  """
    try:
        temp_dir = files.TemporaryDirectory()
        path = temp_dir.__enter__()
    except OSError:
        temp_dir = None
        path = default_dir
    try:
        yield path
    finally:
        if temp_dir:
            temp_dir.__exit__(*sys.exc_info())