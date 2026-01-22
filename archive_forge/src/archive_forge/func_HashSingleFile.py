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
@staticmethod
def HashSingleFile(input_path, algorithm=hashlib.sha256):
    """Gets the hex digest of a single file.

    Args:
      input_path: str, The file path of the contents to add.
      algorithm: a hashing algorithm method, ala hashlib.algorithms

    Returns:
      str, The checksum digest of the file as a hex string.
    """
    return Checksum.FromSingleFile(input_path, algorithm=algorithm).HexDigest()