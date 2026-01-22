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
def HasWriteAccessInDir(directory):
    """Determines if the current user is able to modify the contents of the dir.

  Args:
    directory: str, The full path of the directory to check.

  Raises:
    ValueError: If the given directory path is not a valid directory.

  Returns:
    True if the current user has missing write and execute permissions.
  """
    if not os.path.isdir(directory):
        raise ValueError('The given path [{path}] is not a directory.'.format(path=directory))
    path = os.path.join(directory, '.')
    if not os.access(path, os.X_OK) or not os.access(path, os.W_OK):
        return False
    path = os.path.join(directory, '.HasWriteAccessInDir{pid}'.format(pid=os.getpid()))
    for _ in range(10):
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 438)
            os.close(fd)
        except OSError as e:
            if e.errno == errno.EACCES:
                return False
            if e.errno in [errno.ENOTDIR, errno.ENOENT]:
                raise ValueError('The given path [{path}] is not a directory.'.format(path=directory))
            raise
        try:
            os.remove(path)
            return True
        except OSError as e:
            if e.errno == errno.EACCES:
                return False
            if e.errno != errno.ENOENT:
                raise
    return False