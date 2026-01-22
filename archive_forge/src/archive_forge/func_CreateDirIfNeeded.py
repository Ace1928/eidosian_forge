from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import locale
import os
import struct
import sys
import six
from gslib.utils.constants import WINDOWS_1252
def CreateDirIfNeeded(dir_path, mode=511):
    """Creates a directory, suppressing already-exists errors."""
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode)
        except OSError as e:
            if e.errno != errno.EEXIST and e.errno != errno.EISDIR:
                raise