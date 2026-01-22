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
class OpenSSLCipherTests(TestCase):
    """
    Tests for twisted.internet._sslverify.OpenSSLCipher.
    """
    if skipSSL:
        skip = skipSSL
    cipherName = 'CIPHER-STRING'

    def test_constructorSetsFullName(self):
        """
        The first argument passed to the constructor becomes the full name.
        """
        self.assertEqual(self.cipherName, sslverify.OpenSSLCipher(self.cipherName).fullName)

    def test_repr(self):
        """
        C{repr(cipher)} returns a valid constructor call.
        """
        cipher = sslverify.OpenSSLCipher(self.cipherName)
        self.assertEqual(cipher, eval(repr(cipher), {'OpenSSLCipher': sslverify.OpenSSLCipher}))

    def test_eqSameClass(self):
        """
        Equal type and C{fullName} means that the objects are equal.
        """
        cipher1 = sslverify.OpenSSLCipher(self.cipherName)
        cipher2 = sslverify.OpenSSLCipher(self.cipherName)
        self.assertEqual(cipher1, cipher2)

    def test_eqSameNameDifferentType(self):
        """
        If ciphers have the same name but different types, they're still
        different.
        """

        class DifferentCipher:
            fullName = self.cipherName
        self.assertNotEqual(sslverify.OpenSSLCipher(self.cipherName), DifferentCipher())