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
def _process_Notify(self, r, w):
    reads = self.reads
    writes = self.writes
    _drdw = self._doReadOrWrite
    _logrun = log.callWithLogger
    for selectables, method, dct in ((r, 'doRead', reads), (w, 'doWrite', writes)):
        for selectable in selectables:
            if selectable not in dct:
                continue
            _logrun(selectable, _drdw, selectable, method, dct)