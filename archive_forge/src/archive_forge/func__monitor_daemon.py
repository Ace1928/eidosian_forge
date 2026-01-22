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
def _monitor_daemon(daemon_pid):
    last_restart = None
    while True:
        retval, status = _waitpid(daemon_pid, 0)
        if retval < 0:
            sys.stderr.write('waitpid failed\n')
            sys.exit(1)
        elif retval == daemon_pid:
            status_msg = 'pid %d died, %s' % (daemon_pid, ovs.process.status_msg(status))
            if _should_restart(status):
                if sys.platform != 'win32' and os.WCOREDUMP(status):
                    try:
                        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                    except resource.error:
                        vlog.warn('failed to disable core dumps')
                if last_restart is not None and ovs.timeval.msec() < last_restart + 10000:
                    vlog.warn('%s, waiting until 10 seconds since last restart' % status_msg)
                    while True:
                        now = ovs.timeval.msec()
                        wakeup = last_restart + 10000
                        if now > wakeup:
                            break
                        sys.stdout.write('sleep %f\n' % ((wakeup - now) / 1000.0))
                        time.sleep((wakeup - now) / 1000.0)
                last_restart = ovs.timeval.msec()
                vlog.err('%s, restarting' % status_msg)
                daemon_pid = _fork_and_wait_for_startup()
                if not daemon_pid:
                    break
            else:
                vlog.info('%s, exiting' % status_msg)
                sys.exit(0)