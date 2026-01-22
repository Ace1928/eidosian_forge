import base64
import hmac
import itertools
from collections import OrderedDict
from hashlib import md5
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.portal
import twisted.internet.protocol
import twisted.mail.pop3
import twisted.mail.protocols
from twisted import cred, internet, mail
from twisted.cred.credentials import IUsernameHashedPassword
from twisted.internet import defer
from twisted.internet.testing import LineSendingProtocol
from twisted.mail import pop3
from twisted.protocols import loopback
from twisted.python import failure
from twisted.trial import unittest, util
class CapabilityTests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}'s per-user capability handling.
    """

    def setUp(self):
        """
        Create a POP3 server with some capabilities.
        """
        s = BytesIO()
        p = pop3.POP3()
        p.factory = TestServerFactory()
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        p.do_CAPA()
        self.caps = p.listCapabilities()
        self.pcaps = s.getvalue().splitlines()
        s = BytesIO()
        p.mbox = TestMailbox()
        p.transport = internet.protocol.FileWrapper(s)
        p.do_CAPA()
        self.lpcaps = s.getvalue().splitlines()
        p.connectionLost(failure.Failure(Exception('Test harness disconnect')))

    def test_UIDL(self):
        """
        The server can advertise the I{UIDL} capability.
        """
        contained(self, b'UIDL', self.caps, self.pcaps, self.lpcaps)

    def test_TOP(self):
        """
        The server can advertise the I{TOP} capability.
        """
        contained(self, b'TOP', self.caps, self.pcaps, self.lpcaps)

    def test_USER(self):
        """
        The server can advertise the I{USER} capability.
        """
        contained(self, b'USER', self.caps, self.pcaps, self.lpcaps)

    def test_EXPIRE(self):
        """
        The server can advertise its per-user expiration as well as a global
        expiration.
        """
        contained(self, b'EXPIRE 60 USER', self.caps, self.pcaps)
        contained(self, b'EXPIRE 25', self.lpcaps)

    def test_IMPLEMENTATION(self):
        """
        The server can advertise its implementation string.
        """
        contained(self, b'IMPLEMENTATION Test Implementation String', self.caps, self.pcaps, self.lpcaps)

    def test_SASL(self):
        """
        The server can advertise the SASL schemes it supports.
        """
        contained(self, b'SASL SCHEME_1 SCHEME_2', self.caps, self.pcaps, self.lpcaps)

    def test_LOGIN_DELAY(self):
        """
        The can advertise a per-user login delay as well as a global login
        delay.
        """
        contained(self, b'LOGIN-DELAY 120 USER', self.caps, self.pcaps)
        self.assertIn(b'LOGIN-DELAY 100', self.lpcaps)