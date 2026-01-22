from ctypes import (
import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sysconfig
from .exception import ArchiveError
def archive_error(archive_p, retcode):
    msg = _error_string(archive_p)
    return ArchiveError(msg, errno(archive_p), retcode, archive_p)