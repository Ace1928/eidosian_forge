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
class ClientOptionsTests(SynchronousTestCase):
    """
    Tests for L{sslverify.optionsForClientTLS}.
    """
    if skipSSL:
        skip = skipSSL

    def test_extraKeywords(self):
        """
        When passed a keyword parameter other than C{extraCertificateOptions},
        L{sslverify.optionsForClientTLS} raises an exception just like a
        normal Python function would.
        """
        error = self.assertRaises(TypeError, sslverify.optionsForClientTLS, hostname='alpha', someRandomThing='beta')
        self.assertEqual(str(error), "optionsForClientTLS() got an unexpected keyword argument 'someRandomThing'")

    def test_bytesFailFast(self):
        """
        If you pass L{bytes} as the hostname to
        L{sslverify.optionsForClientTLS} it immediately raises a L{TypeError}.
        """
        error = self.assertRaises(TypeError, sslverify.optionsForClientTLS, b'not-actually-a-hostname.com')
        expectedText = 'optionsForClientTLS requires text for host names, not ' + bytes.__name__
        self.assertEqual(str(error), expectedText)

    def test_dNSNameHostname(self):
        """
        If you pass a dNSName to L{sslverify.optionsForClientTLS}
        L{_hostnameIsDnsName} will be True
        """
        options = sslverify.optionsForClientTLS('example.com')
        self.assertTrue(options._hostnameIsDnsName)

    def test_IPv4AddressHostname(self):
        """
        If you pass an IPv4 address to L{sslverify.optionsForClientTLS}
        L{_hostnameIsDnsName} will be False
        """
        options = sslverify.optionsForClientTLS('127.0.0.1')
        self.assertFalse(options._hostnameIsDnsName)

    def test_IPv6AddressHostname(self):
        """
        If you pass an IPv6 address to L{sslverify.optionsForClientTLS}
        L{_hostnameIsDnsName} will be False
        """
        options = sslverify.optionsForClientTLS('::1')
        self.assertFalse(options._hostnameIsDnsName)