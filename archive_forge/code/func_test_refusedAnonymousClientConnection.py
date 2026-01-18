import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
def test_refusedAnonymousClientConnection(self):
    """
        Check that anonymous connections are refused when certificates are
        required on the server.
        """
    onServerLost = defer.Deferred()
    onClientLost = defer.Deferred()
    self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=True, caCerts=[self.sCert], requireCertificate=True), sslverify.OpenSSLCertificateOptions(requireCertificate=False), onServerLost=onServerLost, onClientLost=onClientLost)
    d = defer.DeferredList([onClientLost, onServerLost], consumeErrors=True)

    def afterLost(result):
        (cSuccess, cResult), (sSuccess, sResult) = result
        self.assertFalse(cSuccess)
        self.assertFalse(sSuccess)
        self.assertIsInstance(cResult.value, (SSL.Error, ConnectionLost))
        self.assertIsInstance(sResult.value, SSL.Error)
    return d.addCallback(afterLost)