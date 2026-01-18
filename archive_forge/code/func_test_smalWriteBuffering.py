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
def test_smalWriteBuffering(self):
    """
        If a small amount data is written to the TLS transport, it is only
        delivered if time passes, indicating small-write buffering is in
        effect.
        """
    client, server, pump = handshakingClientAndServer()
    wrappedServerProtocol = server.wrappedProtocol
    pump.flush()
    self.assertEqual(wrappedServerProtocol.received, [])
    client.write(b'hel')
    client.write(b'lo')
    self.assertEqual(wrappedServerProtocol.received, [])
    pump.flush(advanceClock=False)
    self.assertEqual(wrappedServerProtocol.received, [])
    pump.flush(advanceClock=True)
    self.assertEqual(b''.join(wrappedServerProtocol.received), b'hello')