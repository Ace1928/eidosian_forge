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
def test_openSSL110(self):
    """
        No configuration of contexts occurs under OpenSSL 1.1.0 and
        later, because they create contexts with secure ECDH curves.

        @see: U{http://twistedmatrix.com/trac/ticket/9210}
        """
    chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_110, openSSLlib=self.lib, openSSLcrypto=self.crypto)
    chooser.configureECDHCurve(self.context)
    self.assertFalse(self.libState.ecdhContexts)
    self.assertFalse(self.libState.ecdhValues)
    self.assertFalse(self.cryptoState.getEllipticCurveCalls)
    self.assertIsNone(self.context._ecCurve)