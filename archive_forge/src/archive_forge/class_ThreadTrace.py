from __future__ import print_function
import contextlib
import cProfile
import gc
import inspect
import os
import re
import sys
import threading
import time
import traceback
import types
import warnings
import weakref
from time import perf_counter
from numpy import ndarray
from .Qt import QT_LIB, QtCore
from .util import cprint
from .util.mutex import Mutex
class ThreadTrace(object):
    """ 
    Used to debug freezing by starting a new thread that reports on the 
    location of other threads periodically.
    """

    def __init__(self, interval=10.0, logFile=None):
        self.interval = interval
        self.lock = Mutex()
        self._stop = False
        self.logFile = logFile
        self.start()

    def stop(self):
        with self.lock:
            self._stop = True

    def start(self, interval=None):
        if interval is not None:
            self.interval = interval
        self._stop = False
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        iter = 0
        with open_maybe_console(self.logFile) as printFile:
            while True:
                with self.lock:
                    if self._stop is True:
                        return
                printFile.write(f'\n=============  THREAD FRAMES {iter}:  ================\n')
                for id, frame in sys._current_frames().items():
                    if id == threading.current_thread().ident:
                        continue
                    name = threadName()
                    printFile.write('<< thread %d "%s" >>\n' % (id, name))
                    tb = str(''.join(traceback.format_stack(frame)))
                    printFile.write(tb)
                    printFile.write('\n')
                printFile.write('===============================================\n\n')
                printFile.flush()
                iter += 1
                time.sleep(self.interval)