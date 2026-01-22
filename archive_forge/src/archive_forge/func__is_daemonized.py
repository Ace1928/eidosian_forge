import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def _is_daemonized(self):
    """Return boolean indicating if the current process is
        running as a daemon.

        The criteria to determine the `daemon` condition is to verify
        if the current pid is not the same as the one that got used on
        the initial construction of the plugin *and* the stdin is not
        connected to a terminal.

        The sole validation of the tty is not enough when the plugin
        is executing inside other process like in a CI tool
        (Buildbot, Jenkins).
        """
    return self._original_pid != os.getpid() and (not os.isatty(sys.stdin.fileno()))