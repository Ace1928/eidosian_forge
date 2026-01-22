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
def _fork_and_wait_for_startup():
    if sys.platform == 'win32':
        return _fork_and_wait_for_startup_windows()
    try:
        rfd, wfd = os.pipe()
    except OSError as e:
        sys.stderr.write('pipe failed: %s\n' % os.strerror(e.errno))
        sys.exit(1)
    try:
        pid = os.fork()
    except OSError as e:
        sys.stderr.write('could not fork: %s\n' % os.strerror(e.errno))
        sys.exit(1)
    if pid > 0:
        os.close(wfd)
        ovs.fatal_signal.fork()
        while True:
            try:
                s = os.read(rfd, 1)
                error = 0
            except OSError as e:
                s = ''
                error = e.errno
            if error != errno.EINTR:
                break
        if len(s) != 1:
            retval, status = _waitpid(pid, 0)
            if retval == pid:
                if os.WIFEXITED(status) and os.WEXITSTATUS(status):
                    sys.exit(os.WEXITSTATUS(status))
                else:
                    sys.stderr.write('fork child failed to signal startup (%s)\n' % ovs.process.status_msg(status))
            else:
                assert retval < 0
                sys.stderr.write('waitpid failed (%s)\n' % os.strerror(-retval))
                sys.exit(1)
        os.close(rfd)
    else:
        os.close(rfd)
        ovs.timeval.postfork()
        global _daemonize_fd
        _daemonize_fd = wfd
    return pid