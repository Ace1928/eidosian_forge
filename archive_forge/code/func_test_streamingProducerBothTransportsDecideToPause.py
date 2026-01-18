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
def test_streamingProducerBothTransportsDecideToPause(self):
    """
        pauseProducing() events can come from both the TLS transport layer and
        the underlying transport. In this case, both decide to pause,
        underlying first.
        """

    class PausingStringTransport(StringTransport):
        _didPause = False

        def write(self, data):
            if not self._didPause and self.producer is not None:
                self._didPause = True
                self.producer.pauseProducing()
            StringTransport.write(self, data)

    class TLSConnection:

        def __init__(self):
            self.l = []

        def send(self, data):
            if not self.l:
                data = data[:-1]
            if len(self.l) == 1:
                self.l.append('paused')
                raise WantReadError()
            self.l.append(data)
            return len(data)

        def set_connect_state(self):
            pass

        def do_handshake(self):
            pass

        def bio_write(self, data):
            pass

        def bio_read(self, size):
            return b'X'

        def recv(self, size):
            raise WantReadError()
    transport = PausingStringTransport()
    clientProtocol, tlsProtocol, producer = self.setupStreamingProducer(transport, fakeConnection=TLSConnection())
    self.assertEqual(producer.producerState, 'producing')
    clientProtocol.transport.write(b'hello')
    tlsProtocol.factory._clock.advance(0)
    self.assertEqual(producer.producerState, 'paused')
    self.assertEqual(producer.producerHistory, ['pause'])
    tlsProtocol.transport.producer.resumeProducing()
    self.assertEqual(producer.producerState, 'producing')
    self.assertEqual(producer.producerHistory, ['pause', 'resume'])
    tlsProtocol.dataReceived(b'hello')
    self.assertEqual(producer.producerState, 'producing')
    self.assertEqual(producer.producerHistory, ['pause', 'resume'])