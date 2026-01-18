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
def test_successfulSymmetricSelfSignedCertificateVerification(self):
    """
        Test a successful connection with validation on both server and client
        sides.
        """
    onData = defer.Deferred()
    self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=True, requireCertificate=True, caCerts=[self.cCert]), sslverify.OpenSSLCertificateOptions(privateKey=self.cKey, certificate=self.cCert, verify=True, requireCertificate=True, caCerts=[self.sCert]), onData=onData)
    return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))