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
def FindExecutableOnPath(executable, path=None, pathext=None, allow_extensions=False):
    """Searches for `executable` in the directories listed in `path` or $PATH.

  Executable must not contain a directory or an extension.

  Args:
    executable: The name of the executable to find.
    path: A list of directories to search separated by 'os.pathsep'.  If None
      then the system PATH is used.
    pathext: An iterable of file name extensions to use.  If None then
      platform specific extensions are used.
    allow_extensions: A boolean flag indicating whether extensions in the
      executable are allowed.

  Returns:
    The path of 'executable' (possibly with a platform-specific extension) if
    found and executable, None if not found.

  Raises:
    ValueError: if executable has a path or an extension, and extensions are
      not allowed, or if there's an internal error.
  """
    if not allow_extensions and os.path.splitext(executable)[1]:
        raise ValueError('FindExecutableOnPath({0},...) failed because first argument must not have an extension.'.format(executable))
    if os.path.dirname(executable):
        raise ValueError('FindExecutableOnPath({0},...) failed because first argument must not have a path.'.format(executable))
    if path is None:
        effective_path = _GetSystemPath()
        if effective_path is None:
            return None
    else:
        effective_path = path
    effective_pathext = pathext if pathext is not None else _PlatformExecutableExtensions(platforms.OperatingSystem.Current())
    return _FindExecutableOnPath(executable, effective_path, effective_pathext)