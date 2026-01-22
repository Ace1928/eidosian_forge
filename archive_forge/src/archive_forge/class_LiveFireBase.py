import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
class LiveFireBase:
    """
    Utility for connected reactor-using tests.
    """

    def setUp(self):
        """
        Create an amp server and connect a client to it.
        """
        from twisted.internet import reactor
        self.serverFactory = protocol.ServerFactory()
        self.serverFactory.protocol = self.serverProto
        self.clientFactory = protocol.ClientFactory()
        self.clientFactory.protocol = self.clientProto
        self.clientFactory.onMade = defer.Deferred()
        self.serverFactory.onMade = defer.Deferred()
        self.serverPort = reactor.listenTCP(0, self.serverFactory)
        self.addCleanup(self.serverPort.stopListening)
        self.clientConn = reactor.connectTCP('127.0.0.1', self.serverPort.getHost().port, self.clientFactory)
        self.addCleanup(self.clientConn.disconnect)

        def getProtos(rlst):
            self.cli = self.clientFactory.theProto
            self.svr = self.serverFactory.theProto
        dl = defer.DeferredList([self.clientFactory.onMade, self.serverFactory.onMade])
        return dl.addCallback(getProtos)

    def tearDown(self):
        """
        Cleanup client and server connections, and check the error got at
        C{connectionLost}.
        """
        L = []
        for conn in (self.cli, self.svr):
            if conn.transport is not None:
                d = defer.Deferred().addErrback(_loseAndPass, conn)
                conn.connectionLost = d.errback
                conn.transport.loseConnection()
                L.append(d)
        return defer.gatherResults(L).addErrback(lambda first: first.value.subFailure)