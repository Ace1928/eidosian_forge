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
def test_unsupported_publickey(self):
    """
        Private key authentication fails when the public key type is
        unsupported or the public key is corrupt.
        """
    blob = keys.Key.fromString(keydata.publicDSA_openssh).blob()
    blob = NS(b'ssh-bad-type') + blob[11:]
    packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + NS(blob)
    d = self.authServer.ssh_USERAUTH_REQUEST(packet)
    return d.addCallback(self._checkFailed)