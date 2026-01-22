import errno
import os
import sys
from stat import S_IMODE, S_ISDIR, ST_MODE
from .. import osutils, transport, urlutils
def _get_append_file(self, relpath, mode=None):
    """Call os.open() for the given relpath"""
    file_abspath = self._abspath(relpath)
    if mode is None:
        local_mode = 438
    else:
        local_mode = mode
    try:
        return (file_abspath, os.open(file_abspath, _append_flags, local_mode))
    except OSError as e:
        self._translate_error(e, relpath)