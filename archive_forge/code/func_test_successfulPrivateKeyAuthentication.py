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
def test_successfulPrivateKeyAuthentication(self):
    """
        Test that private key authentication completes successfully,
        """
    blob = keys.Key.fromString(keydata.publicRSA_openssh).blob()
    obj = keys.Key.fromString(keydata.privateRSA_openssh)
    packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\xff' + NS(obj.sshType()) + NS(blob)
    self.authServer.transport.sessionID = b'test'
    signature = obj.sign(NS(b'test') + bytes((userauth.MSG_USERAUTH_REQUEST,)) + packet)
    packet += NS(signature)
    d = self.authServer.ssh_USERAUTH_REQUEST(packet)

    def check(ignored):
        self.assertEqual(self.authServer.transport.packets, [(userauth.MSG_USERAUTH_SUCCESS, b'')])
    return d.addCallback(check)