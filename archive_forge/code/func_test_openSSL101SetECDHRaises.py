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