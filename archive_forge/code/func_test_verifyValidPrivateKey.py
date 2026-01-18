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
def test_verifyValidPrivateKey(self):
    """
        Test that verifying a valid private key works.
        """
    blob = keys.Key.fromString(keydata.publicRSA_openssh).blob()
    packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + NS(blob)
    d = self.authServer.ssh_USERAUTH_REQUEST(packet)

    def check(ignored):
        self.assertEqual(self.authServer.transport.packets, [(userauth.MSG_USERAUTH_PK_OK, NS(b'ssh-rsa') + NS(blob))])
    return d.addCallback(check)