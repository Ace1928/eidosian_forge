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
def test_unencryptedConnectionWithoutPasswords(self):
    """
        If the L{SSHUserAuthServer} is not advertising passwords, then an
        unencrypted connection should not cause any warnings or exceptions.
        This is a white box test.
        """
    portal = Portal(self.realm)
    portal.registerChecker(PrivateKeyChecker())
    clearAuthServer = userauth.SSHUserAuthServer()
    clearAuthServer.transport = FakeTransport(portal)
    clearAuthServer.transport.isEncrypted = lambda x: False
    clearAuthServer.serviceStarted()
    clearAuthServer.serviceStopped()
    self.assertEqual(clearAuthServer.supportedAuthentications, [b'publickey'])
    halfAuthServer = userauth.SSHUserAuthServer()
    halfAuthServer.transport = FakeTransport(portal)
    halfAuthServer.transport.isEncrypted = lambda x: x == 'in'
    halfAuthServer.serviceStarted()
    halfAuthServer.serviceStopped()
    self.assertEqual(clearAuthServer.supportedAuthentications, [b'publickey'])