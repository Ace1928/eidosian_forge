from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_writeUnicodeRaisesTypeError(self):
    """
        Writing C{unicode} to L{TLSMemoryBIOProtocol} throws a C{TypeError}.
        """
    notBytes = 'hello'
    result = []

    class SimpleSendingProtocol(Protocol):

        def connectionMade(self):
            try:
                self.transport.write(notBytes)
                self.transport.write(b'bytes')
                self.transport.loseConnection()
            except TypeError:
                result.append(True)
                self.transport.abortConnection()

    def flush_logged_errors():
        self.assertEqual(len(self.flushLoggedErrors(ConnectionLost, TypeError)), 2)
    d = self.writeBeforeHandshakeTest(SimpleSendingProtocol, b'bytes')
    d.addBoth(lambda ign: self.assertEqual(result, [True]))
    d.addBoth(lambda ign: deferLater(reactor, 0, flush_logged_errors))
    return d