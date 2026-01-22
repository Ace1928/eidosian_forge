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
def _open_brz_log():
    """Open the brz.log trace file.

    If the log is more than a particular length, the old file is renamed to
    brz.log.old and a new file is started.  Otherwise, we append to the
    existing file.

    This sets the global _brz_log_filename.
    """
    global _brz_log_filename

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
    try:
        _brz_log_filename = _get_brz_log_filename()
        _rollover_trace_maybe(_brz_log_filename)
        brz_log_file = _open_or_create_log_file(_brz_log_filename)
        brz_log_file.write(b'\n')
        if brz_log_file.tell() <= 2:
            brz_log_file.write(b'this is a debug log for diagnosing/reporting problems in brz\n')
            brz_log_file.write(b'you can delete or truncate this file, or include sections in\n')
            brz_log_file.write(b'bug reports to https://bugs.launchpad.net/brz/+filebug\n\n')
        return brz_log_file
    except OSError as e:
        sys.stderr.write('failed to open trace file: {}\n'.format(e))
    return None