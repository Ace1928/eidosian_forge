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
class ModuleInitializationTests(unittest.TestCase):
    if keys is None:
        skip = 'cannot run without cryptography'

    def test_messages(self):
        self.assertEqual(userauth.SSHUserAuthServer.protocolMessages[60], 'MSG_USERAUTH_PK_OK')
        self.assertEqual(userauth.SSHUserAuthClient.protocolMessages[60], 'MSG_USERAUTH_PK_OK')