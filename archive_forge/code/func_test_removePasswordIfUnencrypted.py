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
def test_removePasswordIfUnencrypted(self):
    """
        Test that the userauth service does not advertise password
        authentication if the password would be send in cleartext.
        """
    self.assertIn(b'password', self.authServer.supportedAuthentications)
    clearAuthServer = userauth.SSHUserAuthServer()
    clearAuthServer.transport = FakeTransport(self.portal)
    clearAuthServer.transport.isEncrypted = lambda x: False
    clearAuthServer.serviceStarted()
    clearAuthServer.serviceStopped()
    self.assertNotIn(b'password', clearAuthServer.supportedAuthentications)
    halfAuthServer = userauth.SSHUserAuthServer()
    halfAuthServer.transport = FakeTransport(self.portal)
    halfAuthServer.transport.isEncrypted = lambda x: x == 'in'
    halfAuthServer.serviceStarted()
    halfAuthServer.serviceStopped()
    self.assertIn(b'password', halfAuthServer.supportedAuthentications)