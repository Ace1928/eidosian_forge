from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def _open_or_create_log_file(filename):
    """Open existing log file, or create with ownership and permissions

        It inherits the ownership and permissions (masked by umask) from
        the containing directory to cope better with being run under sudo
        with $HOME still set to the user's homedir.
        """
    flags = os.O_WRONLY | os.O_APPEND | osutils.O_TEXT
    while True:
        try:
            fd = os.open(filename, flags)
            break
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
        try:
            fd = os.open(filename, flags | os.O_CREAT | os.O_EXCL, 438)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        else:
            osutils.copy_ownership_from_path(filename)
            break
    return os.fdopen(fd, 'ab', 0)