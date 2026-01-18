import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_rebind(self):
    """
        Re-listening with the same L{DatagramProtocol} re-invokes the
        C{startProtocol} callback.
        """
    server = Server()
    d = server.startedDeferred = defer.Deferred()
    p = reactor.listenUDP(0, server, interface='127.0.0.1')

    def cbStarted(ignored, port):
        return port.stopListening()

    def cbStopped(ignored):
        d = server.startedDeferred = defer.Deferred()
        p = reactor.listenUDP(0, server, interface='127.0.0.1')
        return d.addCallback(cbStarted, p)
    return d.addCallback(cbStarted, p)