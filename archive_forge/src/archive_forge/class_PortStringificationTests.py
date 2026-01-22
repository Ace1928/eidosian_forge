import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class PortStringificationTests(TestCase):

    @skipIf(not interfaces.IReactorTCP(reactor, None), 'IReactorTCP is needed')
    def testTCP(self):
        p = reactor.listenTCP(0, protocol.ServerFactory())
        portNo = p.getHost().port
        self.assertNotEqual(str(p).find(str(portNo)), -1, '%d not found in %s' % (portNo, p))
        return p.stopListening()

    @skipIf(not interfaces.IReactorUDP(reactor, None), 'IReactorUDP is needed')
    def testUDP(self):
        p = reactor.listenUDP(0, protocol.DatagramProtocol())
        portNo = p.getHost().port
        self.assertNotEqual(str(p).find(str(portNo)), -1, '%d not found in %s' % (portNo, p))
        return p.stopListening()

    @skipIf(not interfaces.IReactorSSL(reactor, None), 'IReactorSSL is needed')
    @skipIf(not ssl, 'SSL support is missing')
    def testSSL(self, ssl=ssl):
        pem = util.sibpath(__file__, 'server.pem')
        p = reactor.listenSSL(0, protocol.ServerFactory(), ssl.DefaultOpenSSLContextFactory(pem, pem))
        portNo = p.getHost().port
        self.assertNotEqual(str(p).find(str(portNo)), -1, '%d not found in %s' % (portNo, p))
        return p.stopListening()