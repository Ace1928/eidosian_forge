from __future__ import absolute_import
import errno
import os
import time
from . import (LockBase, AlreadyLocked, LockFailed, NotLocked, NotMyLock,
def write_pid_to_pidfile(pidfile_path):
    """ Write the PID in the named PID file.

        Get the numeric process ID (“PID”) of the current process
        and write it to the named file as a line of text.

        """
    open_flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    open_mode = 420
    pidfile_fd = os.open(pidfile_path, open_flags, open_mode)
    pidfile = os.fdopen(pidfile_fd, 'w')
    pid = os.getpid()
    pidfile.write('%s\n' % pid)
    pidfile.close()