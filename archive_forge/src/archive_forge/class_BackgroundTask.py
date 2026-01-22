import os
import re
import signal as _signal
import sys
import time
import threading
import _thread
from cherrypy._cpcompat import text_or_bytes
from cherrypy._cpcompat import ntob
class BackgroundTask(threading.Thread):
    """A subclass of threading.Thread whose run() method repeats.

    Use this class for most repeating tasks. It uses time.sleep() to wait
    for each interval, which isn't very responsive; that is, even if you call
    self.cancel(), you'll have to wait until the sleep() call finishes before
    the thread stops. To compensate, it defaults to being daemonic, which means
    it won't delay stopping the whole process.
    """

    def __init__(self, interval, function, args=[], kwargs={}, bus=None):
        super(BackgroundTask, self).__init__()
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.running = False
        self.bus = bus
        self.daemon = True

    def cancel(self):
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            time.sleep(self.interval)
            if not self.running:
                return
            try:
                self.function(*self.args, **self.kwargs)
            except Exception:
                if self.bus:
                    self.bus.log('Error in background task thread function %r.' % self.function, level=40, traceback=True)
                raise