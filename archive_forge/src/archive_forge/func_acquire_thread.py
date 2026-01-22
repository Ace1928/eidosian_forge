import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
def acquire_thread(self):
    """Run 'start_thread' listeners for the current thread.

        If the current thread has already been seen, any 'start_thread'
        listeners will not be run again.
        """
    thread_ident = _thread.get_ident()
    if thread_ident not in self.threads:
        i = len(self.threads) + 1
        self.threads[thread_ident] = i
        self.bus.publish('start_thread', i)