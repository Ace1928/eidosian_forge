from __future__ import (absolute_import, division, print_function)
import time
import threading
from ansible.plugins.callback import CallbackBase
class MemProf(threading.Thread):
    """Python thread for recording memory usage"""

    def __init__(self, path, obj=None):
        threading.Thread.__init__(self)
        self.obj = obj
        self.path = path
        self.results = []
        self.running = True

    def run(self):
        while self.running:
            with open(self.path) as f:
                val = f.read()
            self.results.append(int(val.strip()) / 1024 / 1024)
            time.sleep(0.001)