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
def ReadStdinBytes():
    """Reads raw bytes from sys.stdin without any encoding interpretation.

  Returns:
    bytes, The byte string that was read.
  """
    if six.PY2:
        with _FileInBinaryMode(sys.stdin):
            return sys.stdin.read()
    else:
        return sys.stdin.buffer.read()