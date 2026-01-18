import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_removeAllSkipsInternalReaders(self):
    """
        Any L{IReadDescriptor}s in L{PosixReactorBase._internalReaders} are
        left alone by L{PosixReactorBase._removeAll}.
        """
    reactor = TrivialReactor()
    extra = object()
    reactor._internalReaders.add(extra)
    reactor.addReader(extra)
    reactor._removeAll(reactor._readers, reactor._writers)
    self._checkWaker(reactor)
    self.assertIn(extra, reactor._internalReaders)
    self.assertIn(extra, reactor._readers)