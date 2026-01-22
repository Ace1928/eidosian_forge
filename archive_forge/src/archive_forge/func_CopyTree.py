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
def CopyTree(src, dst):
    """Copies a directory recursively, without copying file stat info.

  More specifically, behaves like `cp -R` rather than `cp -Rp`, which means that
  the destination directory and its contents will be *writable* and *deletable*.

  (Yes, an omnipotent being can shutil.copytree a directory so read-only that
  they cannot delete it. But they cannot do that with this function.)

  Adapted from shutil.copytree.

  Args:
    src: str, the path to the source directory
    dst: str, the path to the destination directory. Must not already exist and
      be writable.

  Raises:
    shutil.Error: if copying failed for any reason.
  """
    os.makedirs(dst)
    errors = []
    for name in os.listdir(src):
        name = encoding_util.Decode(name)
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.isdir(srcname):
                CopyTree(srcname, dstname)
            else:
                shutil.copy2(srcname, dstname)
        except shutil.Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, six.text_type(why)))
    if errors:
        raise shutil.Error(errors)