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
def test_openSSL101NoECC(self):
    """
        Contexts created under an OpenSSL 1.0.1 that doesn't support
        ECC have no configuration applied.
        """
    self.cryptoState.getEllipticCurveRaises = ValueError
    chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_101, openSSLlib=self.lib, openSSLcrypto=self.crypto)
    chooser.configureECDHCurve(self.context)
    self.assertFalse(self.libState.ecdhContexts)
    self.assertFalse(self.libState.ecdhValues)
    self.assertIsNone(self.context._ecCurve)