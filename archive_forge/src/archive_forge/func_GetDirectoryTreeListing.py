from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import errno
import hashlib
import io
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import encoding as encoding_util
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import retry
import six
from six.moves import range  # pylint: disable=redefined-builtin
def GetDirectoryTreeListing(path, include_dirs=False, file_predicate=None, dir_sort_func=None, file_sort_func=None):
    """Yields a generator that list all the files in a directory tree.

  Walks directory tree from path and yeilds all files that it finds. Will expand
  paths relative to home dir e.g. those that start with '~'.

  Args:
    path: string, base of file tree to walk.
    include_dirs: bool, if true will yield directory names in addition to files.
    file_predicate: function, boolean function to determine which files should
      be included in the output. Default is all files.
    dir_sort_func: function, function that will determine order directories are
      processed. Default is lexical ordering.
    file_sort_func:  function, function that will determine order directories
      are processed. Default is lexical ordering.
  Yields:
    Generator: yields all files and directory paths matching supplied criteria.
  """
    if not file_sort_func:
        file_sort_func = sorted
    if file_predicate is None:
        file_predicate = lambda x: True
    if dir_sort_func is None:
        dir_sort_func = lambda x: x.sort()
    for root, dirs, files in os.walk(ExpandHomeDir(six.text_type(path))):
        dir_sort_func(dirs)
        if include_dirs:
            for dirname in dirs:
                yield dirname
        for file_name in file_sort_func(files):
            file_path = os.path.join(root, file_name)
            if file_predicate(file_path):
                yield file_path