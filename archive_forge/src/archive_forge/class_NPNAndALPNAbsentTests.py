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
class NPNAndALPNAbsentTests(TestCase):
    """
    NPN/ALPN operations fail on platforms that do not support them.

    These tests only run on platforms that have a PyOpenSSL version < 0.15,
    an OpenSSL version earlier than 1.0.1, or an OpenSSL/cryptography built
    without NPN support.
    """
    if skipSSL:
        skip = skipSSL
    elif not skipNPN or not skipALPN:
        skip = 'NPN and/or ALPN is present on this platform'

    def test_nextProtocolMechanismsNoNegotiationSupported(self):
        """
        When neither NPN or ALPN are available on a platform, there are no
        supported negotiation protocols.
        """
        supportedProtocols = sslverify.protocolNegotiationMechanisms()
        self.assertFalse(supportedProtocols)

    def test_NPNAndALPNNotImplemented(self):
        """
        A NotImplementedError is raised when using acceptableProtocols on a
        platform that does not support either NPN or ALPN.
        """
        protocols = [b'h2', b'http/1.1']
        self.assertRaises(NotImplementedError, negotiateProtocol, serverProtocols=protocols, clientProtocols=protocols)

    def test_NegotiatedProtocolReturnsNone(self):
        """
        negotiatedProtocol return L{None} even when NPN/ALPN aren't supported.
        This works because, as neither are supported, negotiation isn't even
        attempted.
        """
        serverProtocols = None
        clientProtocols = None
        negotiatedProtocol, lostReason = negotiateProtocol(clientProtocols=clientProtocols, serverProtocols=serverProtocols)
        self.assertIsNone(negotiatedProtocol)
        self.assertIsNone(lostReason)