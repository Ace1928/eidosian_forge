from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def _testStreamingProducer(self, mode: Literal['server', 'client']) -> None:
    """
        Connect a couple protocol/transport pairs to an L{IOPump} and then pump
        it.  Verify that a streaming producer registered with one of the
        transports does not receive invalid L{IPushProducer} method calls and
        ends in the right state.

        @param mode: C{u"server"} to test a producer registered with the
            server transport.  C{u"client"} to test a producer registered with
            the client transport.
        """
    serverProto = Protocol()
    serverTransport = FakeTransport(serverProto, isServer=True)
    clientProto = Protocol()
    clientTransport = FakeTransport(clientProto, isServer=False)
    pump = connect(serverProto, serverTransport, clientProto, clientTransport, greet=False)
    producer = StrictPushProducer()
    victim = {'server': serverTransport, 'client': clientTransport}[mode]
    victim.registerProducer(producer, streaming=True)
    pump.pump()
    self.assertEqual('running', producer._state)