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
def ReadBinaryFileContents(path):
    """Reads the binary contents from the given path.

  Args:
    path: str, The file path to read.

  Raises:
    Error: If the file cannot be read.

  Returns:
    bytes, The byte string read from the file.
  """
    try:
        with BinaryFileReader(path) as f:
            return f.read()
    except EnvironmentError as e:
        raise Error('Unable to read file [{0}]: {1}'.format(path, e))