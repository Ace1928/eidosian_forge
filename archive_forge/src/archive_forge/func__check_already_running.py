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
def _check_already_running():
    pid = __read_pidfile(_pidfile, True)
    if pid > 0:
        _fatal('%s: already running as pid %d, aborting' % (_pidfile, pid))
    elif pid < 0:
        _fatal('%s: pidfile check failed (%s), aborting' % (_pidfile, os.strerror(pid)))