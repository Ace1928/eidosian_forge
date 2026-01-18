from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_tryAuthEdgeCases(self):
    """
        tryAuth() has two edge cases that are difficult to reach.

        1) an authentication method auth_* returns None instead of a Deferred.
        2) an authentication type that is defined does not have a matching
           auth_* method.

        Both these cases should return a Deferred which fails with a
        ConchError.
        """

    def mockAuth(packet):
        return None
    self.patch(self.authServer, 'auth_publickey', mockAuth)
    self.patch(self.authServer, 'auth_password', None)

    def secondTest(ignored):
        d2 = self.authServer.tryAuth(b'password', None, None)
        return self.assertFailure(d2, ConchError)
    d1 = self.authServer.tryAuth(b'publickey', None, None)
    return self.assertFailure(d1, ConchError).addCallback(secondTest)