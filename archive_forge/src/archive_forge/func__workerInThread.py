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
def _workerInThread(self):
    try:
        while 1:
            fn, args = self.toThreadQueue.get()
            fn(*args)
    except SystemExit:
        pass
    except BaseException:
        f = failure.Failure()
        self._sendToMain('Failure', f)