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
class GlobalCapabilitiesTests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}'s global capability handling.
    """

    def setUp(self):
        """
        Create a POP3 server with some capabilities.
        """
        s = BytesIO()
        p = pop3.POP3()
        p.factory = TestServerFactory()
        p.factory.pue = p.factory.puld = False
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

    def test_EXPIRE(self):
        """
        I{EXPIRE} is in the server's advertised capabilities.
        """
        contained(self, b'EXPIRE 60', self.caps, self.pcaps, self.lpcaps)

    def test_LOGIN_DELAY(self):
        """
        I{LOGIN-DELAY} is in the server's advertised capabilities.
        """
        contained(self, b'LOGIN-DELAY 120', self.caps, self.pcaps, self.lpcaps)