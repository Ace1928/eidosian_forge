import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
class Daemonizer(SimplePlugin):
    """Daemonize the running script.

    Use this with a Web Site Process Bus via::

        Daemonizer(bus).subscribe()

    When this component finishes, the process is completely decoupled from
    the parent environment. Please note that when this component is used,
    the return code from the parent process will still be 0 if a startup
    error occurs in the forked children. Errors in the initial daemonizing
    process still return proper exit codes. Therefore, if you use this
    plugin to daemonize, don't use the return code as an accurate indicator
    of whether the process fully started. In fact, that return code only
    indicates if the process successfully finished the first fork.
    """

    def __init__(self, bus, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
        SimplePlugin.__init__(self, bus)
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.finalized = False

    def start(self):
        if self.finalized:
            self.bus.log('Already deamonized.')
        if threading.active_count() != 1:
            self.bus.log('There are %r active threads. Daemonizing now may cause strange failures.' % threading.enumerate(), level=30)
        self.daemonize(self.stdin, self.stdout, self.stderr, self.bus.log)
        self.finalized = True
    start.priority = 65

    @staticmethod
    def daemonize(stdin='/dev/null', stdout='/dev/null', stderr='/dev/null', logger=lambda msg: None):
        sys.stdout.flush()
        sys.stderr.flush()
        error_tmpl = '{sys.argv[0]}: fork #{n} failed: ({exc.errno}) {exc.strerror}\n'
        for fork in range(2):
            msg = ['Forking once.', 'Forking twice.'][fork]
            try:
                pid = os.fork()
                if pid > 0:
                    logger(msg)
                    os._exit(0)
            except OSError as exc:
                sys.exit(error_tmpl.format(sys=sys, exc=exc, n=fork + 1))
            if fork == 0:
                os.setsid()
        os.umask(0)
        si = open(stdin, 'r')
        so = open(stdout, 'a+')
        se = open(stderr, 'a+')
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        logger('Daemonized to PID: %s' % os.getpid())