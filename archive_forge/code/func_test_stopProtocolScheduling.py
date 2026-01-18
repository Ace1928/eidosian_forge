import socket
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet import defer, error
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import DatagramProtocol
from twisted.internet.test.connectionmixins import LogObserverMixin, findFreePort
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python import context
from twisted.python.log import ILogContext, err
from twisted.test.test_udp import GoodClient, Server
from twisted.trial.unittest import SkipTest
def test_stopProtocolScheduling(self):
    """
        L{DatagramProtocol.stopProtocol} is called asynchronously (ie, not
        re-entrantly) when C{stopListening} is used to stop the datagram
        transport.
        """

    class DisconnectingProtocol(DatagramProtocol):
        started = False
        stopped = False
        inStartProtocol = False
        stoppedInStart = False

        def startProtocol(self):
            self.started = True
            self.inStartProtocol = True
            self.transport.stopListening()
            self.inStartProtocol = False

        def stopProtocol(self):
            self.stopped = True
            self.stoppedInStart = self.inStartProtocol
            reactor.stop()
    reactor = self.buildReactor()
    protocol = DisconnectingProtocol()
    self.getListeningPort(reactor, protocol)
    self.runReactor(reactor)
    self.assertTrue(protocol.started)
    self.assertTrue(protocol.stopped)
    self.assertFalse(protocol.stoppedInStart)