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
class POP3ClientDeprecationTests(unittest.SynchronousTestCase):
    """
    Tests for the now deprecated L{twisted.mail.pop3client} module.
    """

    def test_deprecation(self):
        """
        A deprecation warning is emitted when directly importing the now
        deprected pop3client module.

        This test might fail is some other code has already imported it.
        No code should use the deprected module.
        """
        from twisted.mail import pop3client
        warningsShown = self.flushWarnings(offendingFunctions=[self.test_deprecation])
        self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], 'twisted.mail.pop3client was deprecated in Twisted 21.2.0. Use twisted.mail.pop3 instead.')
        self.assertEqual(len(warningsShown), 1)
        pop3client