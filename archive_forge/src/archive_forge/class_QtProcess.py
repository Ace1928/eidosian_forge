import atexit
import inspect
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import pickle
from ..Qt import QT_LIB, mkQApp
from ..util import cprint  # color printing for debugging
from .remoteproxy import (
import threading
class QtProcess(Process):
    """
    QtProcess is essentially the same as Process, with two major differences:
    
      - The remote process starts by running startQtEventLoop() which creates a
        QApplication in the remote process and uses a QTimer to trigger
        remote event processing. This allows the remote process to have its own
        GUI.
      - A QTimer is also started on the parent process which polls for requests
        from the child process. This allows Qt signals emitted within the child
        process to invoke slots on the parent process and vice-versa. This can
        be disabled using processRequests=False in the constructor.
      
    Example::
    
        proc = QtProcess()            
        rQtGui = proc._import('PyQt4.QtGui')
        btn = rQtWidgets.QPushButton('button on child process')
        btn.show()
        
        def slot():
            print('slot invoked on parent process')
        btn.clicked.connect(proxy(slot))   # be sure to send a proxy of the slot
    """

    def __init__(self, **kwds):
        if 'target' not in kwds:
            kwds['target'] = startQtEventLoop
        from ..Qt import QtWidgets
        self._processRequests = kwds.pop('processRequests', True)
        if self._processRequests and QtWidgets.QApplication.instance() is None:
            raise Exception('Must create QApplication before starting QtProcess, or use QtProcess(processRequests=False)')
        Process.__init__(self, **kwds)
        self.startEventTimer()

    def startEventTimer(self):
        from ..Qt import QtCore
        self.timer = QtCore.QTimer()
        if self._processRequests:
            self.startRequestProcessing()

    def startRequestProcessing(self, interval=0.01):
        """Start listening for requests coming from the child process.
        This allows signals to be connected from the child process to the parent.
        """
        self.timer.timeout.connect(self.processRequests)
        self.timer.start(int(interval * 1000))

    def stopRequestProcessing(self):
        self.timer.stop()

    def processRequests(self):
        try:
            Process.processRequests(self)
        except ClosedError:
            self.timer.stop()