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
def test_successfulCertificateVerification(self):
    """
        Test a successful connection with client certificate validation on
        server side.
        """
    onData = defer.Deferred()
    self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=False, requireCertificate=False), sslverify.OpenSSLCertificateOptions(verify=True, requireCertificate=True, caCerts=[self.sCert]), onData=onData)
    return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))