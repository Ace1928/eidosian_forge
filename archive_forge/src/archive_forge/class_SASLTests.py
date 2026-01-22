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
class SASLTests(unittest.TestCase):
    """
    Tests for L{pop3.POP3}'s SASL implementation.
    """

    def test_ValidLogin(self):
        """
        A CRAM-MD5-based SASL login attempt succeeds if it uses a username and
        a hashed password known to the server's credentials checker.
        """
        p = pop3.POP3()
        p.factory = TestServerFactory()
        p.factory.challengers = {b'CRAM-MD5': cred.credentials.CramMD5Credentials}
        p.portal = cred.portal.Portal(TestRealm())
        ch = cred.checkers.InMemoryUsernamePasswordDatabaseDontUse()
        ch.addUser(b'testuser', b'testpassword')
        p.portal.registerChecker(ch)
        s = BytesIO()
        p.transport = internet.protocol.FileWrapper(s)
        p.connectionMade()
        p.lineReceived(b'CAPA')
        self.assertTrue(s.getvalue().find(b'SASL CRAM-MD5') >= 0)
        p.lineReceived(b'AUTH CRAM-MD5')
        chal = s.getvalue().splitlines()[-1][2:]
        chal = base64.b64decode(chal)
        response = hmac.HMAC(b'testpassword', chal, digestmod=md5).hexdigest().encode('ascii')
        p.lineReceived(base64.b64encode(b'testuser ' + response))
        self.assertTrue(p.mbox)
        self.assertTrue(s.getvalue().splitlines()[-1].find(b'+OK') >= 0)
        p.connectionLost(failure.Failure(Exception('Test harness disconnect')))