import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_removeAllReturnsRemovedDescriptors(self):
    """
        L{PosixReactorBase._removeAll} returns a list of removed
        L{IReadDescriptor} and L{IWriteDescriptor} objects.
        """
    reactor = TrivialReactor()
    reader = object()
    writer = object()
    reactor.addReader(reader)
    reactor.addWriter(writer)
    removed = reactor._removeAll(reactor._readers, reactor._writers)
    self.assertEqual(set(removed), {reader, writer})
    self.assertNotIn(reader, reactor._readers)
    self.assertNotIn(writer, reactor._writers)