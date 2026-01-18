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
def test_pause(self):
    """
        When the streaming producer is paused, the underlying producer stops
        getting resumeProducing calls.
        """

    class PausingStringTransport(StringTransport):
        writes = 0

        def __init__(self):
            StringTransport.__init__(self)
            self.paused = Deferred()

        def write(self, data):
            self.writes += 1
            StringTransport.write(self, data)
            if self.writes == 3:
                self.producer.pauseProducing()
                d = self.paused
                del self.paused
                d.callback(None)
    consumer = PausingStringTransport()
    nsProducer = NonStreamingProducer(consumer)
    streamingProducer = _PullToPush(nsProducer, consumer)
    consumer.registerProducer(streamingProducer, True)

    def shouldNotBeCalled(ignore):
        self.fail('BUG: The producer should not finish!')
    nsProducer.result.addCallback(shouldNotBeCalled)
    done = consumer.paused

    def paused(ignore):
        self.assertEqual(streamingProducer._coopTask._pauseCount, 1)
    done.addCallback(paused)
    streamingProducer.startStreaming()
    return done