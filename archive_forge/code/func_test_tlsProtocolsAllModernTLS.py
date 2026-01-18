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
def test_tlsProtocolsAllModernTLS(self):
    """
        When calling L{sslverify.OpenSSLCertificateOptions} with
        C{insecurelyLowerMinimumTo} set to TLSv1.0 and
        C{lowerMaximumSecurityTo} to TLSv1.2, it will exclude both SSLs and
        the (unreleased) TLSv1.3.
        """
    opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_0, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_2)
    opts._contextFactory = FakeContext
    ctx = opts.getContext()
    options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | opts._OP_NO_TLSv1_3
    self.assertEqual(options, ctx._options & options)