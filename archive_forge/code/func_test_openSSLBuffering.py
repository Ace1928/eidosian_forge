import os
import hamcrest
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import waitUntilAllDisconnected
from twisted.protocols import basic
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test.test_tcp import ProperlyCloseFilesMixin
from twisted.trial.unittest import TestCase
from zope.interface import implementer
def test_openSSLBuffering(self):
    serverProto = self.serverProto = SingleLineServerProtocol()
    clientProto = self.clientProto = RecordingClientProtocol()
    server = protocol.ServerFactory()
    client = self.client = protocol.ClientFactory()
    server.protocol = lambda: serverProto
    client.protocol = lambda: clientProto
    sCTX = ssl.DefaultOpenSSLContextFactory(certPath, certPath)
    cCTX = ssl.ClientContextFactory()
    port = reactor.listenSSL(0, server, sCTX, interface='127.0.0.1')
    self.addCleanup(port.stopListening)
    clientConnector = reactor.connectSSL('127.0.0.1', port.getHost().port, client, cCTX)
    self.addCleanup(clientConnector.disconnect)
    return clientProto.deferred.addCallback(self.assertEqual, b'+OK <some crap>\r\n')