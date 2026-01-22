from errno import EPERM
from socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, AddressFamily, gaierror
from types import FunctionType
from unicodedata import normalize
from unittest import skipIf
from zope.interface import implementer, providedBy, provider
from zope.interface.interface import InterfaceClass
from zope.interface.verify import verifyClass, verifyObject
from twisted import plugins
from twisted.internet import (
from twisted.internet.abstract import isIPv6Address
from twisted.internet.address import (
from twisted.internet.endpoints import StandardErrorBehavior
from twisted.internet.error import ConnectingCancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol
from twisted.internet.stdio import PipeAddress
from twisted.internet.task import Clock
from twisted.internet.testing import (
from twisted.logger import ILogObserver, globalLogPublisher
from twisted.plugin import getPlugins
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.systemd import ListenFDs
from twisted.test.iosim import connectableEndpoint, connectedServerAndClient
from twisted.trial import unittest
@skipIf(skipSSL, skipSSLReason)
class SSLClientStringTests(unittest.TestCase):
    """
    Tests for L{twisted.internet.endpoints.clientFromString} which require SSL.
    """

    def test_ssl(self):
        """
        When passed an SSL strports description, L{clientFromString} returns a
        L{SSL4ClientEndpoint} instance initialized with the values from the
        string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'ssl:host=example.net:port=4321:privateKey=%s:certKey=%s:bindAddress=10.0.0.3:timeout=3:caCertsDir=%s' % (escapedPEMPathName, escapedPEMPathName, escapedCAsPathName))
        self.assertIsInstance(client, endpoints.SSL4ClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._host, 'example.net')
        self.assertEqual(client._port, 4321)
        self.assertEqual(client._timeout, 3)
        self.assertEqual(client._bindAddress, ('10.0.0.3', 0))
        certOptions = client._sslContextFactory
        self.assertIsInstance(certOptions, CertificateOptions)
        self.assertEqual(certOptions.method, TLS_METHOD)
        self.assertTrue(certOptions._options & OP_NO_SSLv3)
        ctx = certOptions.getContext()
        self.assertIsInstance(ctx, ContextType)
        self.assertEqual(Certificate(certOptions.certificate), testCertificate)
        privateCert = PrivateCertificate(certOptions.certificate)
        privateCert._setPrivateKey(KeyPair(certOptions.privateKey))
        self.assertEqual(privateCert, testPrivateCertificate)
        expectedCerts = [Certificate.loadPEM(x.getContent()) for x in [casPath.child('thing1.pem'), casPath.child('thing2.pem')] if x.basename().lower().endswith('.pem')]
        addedCerts = []

        class ListCtx:

            def get_cert_store(self):

                class Store:

                    def add_cert(self, cert):
                        addedCerts.append(cert)
                return Store()
        certOptions.trustRoot._addCACertsToContext(ListCtx())
        self.assertEqual(sorted((Certificate(x) for x in addedCerts), key=lambda cert: cert.digest()), sorted(expectedCerts, key=lambda cert: cert.digest()))

    def test_sslPositionalArgs(self):
        """
        When passed an SSL strports description, L{clientFromString} returns a
        L{SSL4ClientEndpoint} instance initialized with the values from the
        string.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'ssl:example.net:4321:privateKey=%s:certKey=%s:bindAddress=10.0.0.3:timeout=3:caCertsDir=%s' % (escapedPEMPathName, escapedPEMPathName, escapedCAsPathName))
        self.assertIsInstance(client, endpoints.SSL4ClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._host, 'example.net')
        self.assertEqual(client._port, 4321)
        self.assertEqual(client._timeout, 3)
        self.assertEqual(client._bindAddress, ('10.0.0.3', 0))

    def test_sslWithDefaults(self):
        """
        When passed an SSL strports description without extra arguments,
        L{clientFromString} returns a L{SSL4ClientEndpoint} instance
        whose context factory is initialized with default values.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'ssl:example.net:4321')
        self.assertIsInstance(client, endpoints.SSL4ClientEndpoint)
        self.assertIs(client._reactor, reactor)
        self.assertEqual(client._host, 'example.net')
        self.assertEqual(client._port, 4321)
        certOptions = client._sslContextFactory
        self.assertEqual(certOptions.method, TLS_METHOD)
        self.assertIsNone(certOptions.certificate)
        self.assertIsNone(certOptions.privateKey)

    def test_unreadableCertificate(self):
        """
        If a certificate in the directory is unreadable,
        L{endpoints._loadCAsFromDir} will ignore that certificate.
        """

        class UnreadableFilePath(FilePath):

            def getContent(self):
                data = FilePath.getContent(self)
                if data == casPath.child('thing2.pem').getContent():
                    raise OSError(EPERM)
                else:
                    return data
        casPathClone = casPath.child('ignored').parent()
        casPathClone.clonePath = UnreadableFilePath
        self.assertEqual([Certificate(x) for x in endpoints._loadCAsFromDir(casPathClone)._caCerts], [Certificate.loadPEM(casPath.child('thing1.pem').getContent())])

    def test_sslSimple(self):
        """
        When passed an SSL strports description without any extra parameters,
        L{clientFromString} returns a simple non-verifying endpoint that will
        speak SSL.
        """
        reactor = object()
        client = endpoints.clientFromString(reactor, 'ssl:host=simple.example.org:port=4321')
        certOptions = client._sslContextFactory
        self.assertIsInstance(certOptions, CertificateOptions)
        self.assertFalse(certOptions.verify)
        ctx = certOptions.getContext()
        self.assertIsInstance(ctx, ContextType)