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
def WriteFileContents(path, contents, overwrite=True, private=False, create_path=True, newline=None, convert_invalid_windows_characters=False):
    """Writes the given text contents to a file at the given path.

  Args:
    path: str, The file path to write to.
    contents: str, The text string to write.
    overwrite: bool, False to error out if the file already exists.
    private: bool, True to make the file have 0o600 permissions.
    create_path: bool, True to create intermediate directories, if needed.
    newline: str, The line ending style to use, or None to use platform default.
    convert_invalid_windows_characters: bool, Convert invalid Windows path
        characters with an 'unsupported' symbol rather than trigger an OSError
        on Windows (e.g. "file|.txt" -> "file$.txt").

  Raises:
    Error: If the file cannot be written.
  """
    try:
        _CheckOverwrite(path, overwrite)
        with FileWriter(path, private=private, create_path=create_path, newline=newline, convert_invalid_windows_characters=convert_invalid_windows_characters) as f:
            f.write(encoding_util.Decode(contents))
    except EnvironmentError as e:
        raise Error('Unable to write file [{0}]: {1}'.format(path, e))