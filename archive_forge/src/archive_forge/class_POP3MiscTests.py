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
class POP3MiscTests(unittest.SynchronousTestCase):
    """
    Miscellaneous tests more to do with module/package structure than
    anything to do with the Post Office Protocol.
    """

    def test_all(self):
        """
        This test checks that all names listed in
        twisted.mail.pop3.__all__ are actually present in the module.
        """
        mod = twisted.mail.pop3
        for attr in mod.__all__:
            self.assertTrue(hasattr(mod, attr))