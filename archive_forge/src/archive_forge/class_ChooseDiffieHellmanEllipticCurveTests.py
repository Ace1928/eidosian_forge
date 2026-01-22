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
class ChooseDiffieHellmanEllipticCurveTests(SynchronousTestCase):
    """
    Tests for L{sslverify._ChooseDiffieHellmanEllipticCurve}.

    @cvar OPENSSL_110: A version number for OpenSSL 1.1.0

    @cvar OPENSSL_102: A version number for OpenSSL 1.0.2

    @cvar OPENSSL_101: A version number for OpenSSL 1.0.1

    @see:
        U{https://wiki.openssl.org/index.php/Manual:OPENSSL_VERSION_NUMBER(3)}
    """
    if skipSSL:
        skip = skipSSL
    OPENSSL_110 = 269484159
    OPENSSL_102 = 268443887
    OPENSSL_101 = 268439887

    def setUp(self):
        self.libState = FakeLibState(setECDHAutoRaises=False)
        self.lib = FakeLib(self.libState)
        self.cryptoState = FakeCryptoState(getEllipticCurveReturns=None, getEllipticCurveRaises=None)
        self.crypto = FakeCrypto(self.cryptoState)
        self.context = FakeContext(SSL.SSLv23_METHOD)

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

    def test_openSSL102(self):
        """
        OpenSSL 1.0.2 does not set ECDH curves by default, but
        C{SSL_CTX_set_ecdh_auto} requests that a context choose a
        secure set curves automatically.
        """
        context = SSL.Context(SSL.SSLv23_METHOD)
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_102, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(context)
        self.assertEqual(self.libState.ecdhContexts, [context._context])
        self.assertEqual(self.libState.ecdhValues, [True])
        self.assertFalse(self.cryptoState.getEllipticCurveCalls)
        self.assertIsNone(self.context._ecCurve)

    def test_openSSL102SetECDHAutoRaises(self):
        """
        An exception raised by C{SSL_CTX_set_ecdh_auto} under OpenSSL
        1.0.2 is suppressed because ECDH is best-effort.
        """
        self.libState.setECDHAutoRaises = BaseException
        context = SSL.Context(SSL.SSLv23_METHOD)
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_102, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(context)
        self.assertEqual(self.libState.ecdhContexts, [context._context])
        self.assertEqual(self.libState.ecdhValues, [True])
        self.assertFalse(self.cryptoState.getEllipticCurveCalls)

    def test_openSSL101(self):
        """
        OpenSSL 1.0.1 does not set ECDH curves by default, nor does
        it expose L{SSL_CTX_set_ecdh_auto}.  Instead, a single ECDH
        curve can be set with L{OpenSSL.SSL.Context.set_tmp_ecdh}.
        """
        self.cryptoState.getEllipticCurveReturns = curve = 'curve object'
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_101, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(self.context)
        self.assertFalse(self.libState.ecdhContexts)
        self.assertFalse(self.libState.ecdhValues)
        self.assertEqual(self.cryptoState.getEllipticCurveCalls, [sslverify._defaultCurveName])
        self.assertIs(self.context._ecCurve, curve)

    def test_openSSL101SetECDHRaises(self):
        """
        An exception raised by L{OpenSSL.SSL.Context.set_tmp_ecdh}
        under OpenSSL 1.0.1 is suppressed because ECHDE is best-effort.
        """

        def set_tmp_ecdh(ctx):
            raise BaseException
        self.context.set_tmp_ecdh = set_tmp_ecdh
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_101, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(self.context)
        self.assertFalse(self.libState.ecdhContexts)
        self.assertFalse(self.libState.ecdhValues)
        self.assertEqual(self.cryptoState.getEllipticCurveCalls, [sslverify._defaultCurveName])

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