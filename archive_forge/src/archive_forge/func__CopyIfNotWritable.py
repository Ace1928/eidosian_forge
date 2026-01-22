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
def _CopyIfNotWritable(source_dir, temp_dir):
    """Returns a writable directory with the same contents as source_dir.

  If source_dir is writable, it is used. Otherwise, a directory 'dest' inside of
  temp_dir is used.

  Args:
    source_dir: str, the directory to (potentially) copy
    temp_dir: str, the path to a writable temporary directory in which to store
      any copied code.

  Returns:
    str, the path to a writable directory with the same contents as source_dir
      (i.e. source_dir, if it's writable, or a copy otherwise).

  Raises:
    UploadFailureError: if the command exits non-zero.
    InvalidSourceDirError: if the source directory is not valid.
  """
    if not os.path.isdir(source_dir):
        raise InvalidSourceDirError(source_dir)
    try:
        writable = files.HasWriteAccessInDir(source_dir)
    except ValueError:
        raise InvalidSourceDirError(source_dir)
    if writable:
        return source_dir
    if files.IsDirAncestorOf(source_dir, temp_dir):
        raise UncopyablePackageError('Cannot copy directory since working directory [{}] is inside of source directory [{}].'.format(temp_dir, source_dir))
    dest_dir = os.path.join(temp_dir, 'dest')
    log.debug('Copying local source tree from [%s] to [%s]', source_dir, dest_dir)
    try:
        files.CopyTree(source_dir, dest_dir)
    except OSError:
        raise UncopyablePackageError('Cannot write to working location [{}]'.format(dest_dir))
    return dest_dir