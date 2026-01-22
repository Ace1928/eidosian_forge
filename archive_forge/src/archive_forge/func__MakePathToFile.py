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
def _MakePathToFile(path, mode=511):
    parent_dir_path, _ = os.path.split(path)
    full_parent_dir_path = os.path.realpath(ExpandHomeDir(parent_dir_path))
    MakeDir(full_parent_dir_path, mode)