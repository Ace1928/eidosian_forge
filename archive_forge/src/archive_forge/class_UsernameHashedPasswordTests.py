from binascii import hexlify, unhexlify
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, error, portal
from twisted.internet import defer
from twisted.python import components
from twisted.python.versions import Version
from twisted.trial import unittest
class UsernameHashedPasswordTests(unittest.TestCase):
    """
    UsernameHashedPassword is a deprecated class that is functionally
    equivalent to UsernamePassword.
    """

    def test_deprecation(self):
        """
        Tests that UsernameHashedPassword is deprecated.
        """
        self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion, 'Use twisted.cred.credentials.UsernamePassword instead.')