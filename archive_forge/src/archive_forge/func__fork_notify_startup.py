import errno
import os
import signal
import sys
import time
import ovs.dirs
import ovs.fatal_signal
import ovs.process
import ovs.socket_util
import ovs.timeval
import ovs.util
import ovs.vlog
def _fork_notify_startup(fd):
    if sys.platform == 'win32':
        _fork_notify_startup_windows(fd)
        return
    if fd is not None:
        error, bytes_written = ovs.socket_util.write_fully(fd, '0')
        if error:
            sys.stderr.write('could not write to pipe\n')
            sys.exit(1)
        os.close(fd)