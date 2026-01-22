import select
import sys
from errno import EBADF, EINTR
from functools import partial
from queue import Empty, Queue
from threading import Thread
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.internet.posixbase import _NO_FILEDESC, _NO_FILENO
from twisted.internet.selectreactor import _select
from twisted.python import failure, log, threadable
def _mainLoopShutdown(self):
    self.mainWaker = None
    if self.workerThread is not None:
        self._sendToThread(raiseException, SystemExit)
        self.wakeUp()
        try:
            while 1:
                msg, args = self.toMainThread.get_nowait()
        except Empty:
            pass
        self.workerThread.join()
        self.workerThread = None
    try:
        while 1:
            fn, args = self.toThreadQueue.get_nowait()
            if fn is self._doIterationInThread:
                log.msg('Iteration is still in the thread queue!')
            elif fn is raiseException and args[0] is SystemExit:
                pass
            else:
                fn(*args)
    except Empty:
        pass