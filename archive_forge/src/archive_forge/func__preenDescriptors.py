import select
import sys
from errno import EBADF, EINTR
from time import sleep
from typing import Type
from zope.interface import implementer
from twisted.internet import posixbase
from twisted.internet.interfaces import IReactorFDSet
from twisted.python import log
from twisted.python.runtime import platformType
def _preenDescriptors(self):
    log.msg('Malformed file descriptor found.  Preening lists.')
    readers = list(self._reads)
    writers = list(self._writes)
    self._reads.clear()
    self._writes.clear()
    for selSet, selList in ((self._reads, readers), (self._writes, writers)):
        for selectable in selList:
            try:
                select.select([selectable], [selectable], [selectable], 0)
            except Exception as e:
                log.msg('bad descriptor %s' % selectable)
                self._disconnectSelectable(selectable, e, False)
            else:
                selSet.add(selectable)