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
def test_password(self):
    """
        Test that the client can authentication with a password.  This
        includes changing the password.
        """
    self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\x00')
    self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\x00' + NS(b'foo')))
    self.authClient.ssh_USERAUTH_PK_OK(NS(b'') + NS(b''))
    self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\xff' + NS(b'foo') * 2))