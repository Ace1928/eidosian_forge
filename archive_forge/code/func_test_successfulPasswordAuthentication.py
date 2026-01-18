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
def test_successfulPasswordAuthentication(self):
    """
        When provided with correct password authentication information, the
        server should respond by sending a MSG_USERAUTH_SUCCESS message with
        no other data.

        See RFC 4252, Section 5.1.
        """
    packet = b''.join([NS(b'foo'), NS(b'none'), NS(b'password'), b'\x00', NS(b'foo')])
    d = self.authServer.ssh_USERAUTH_REQUEST(packet)

    def check(ignored):
        self.assertEqual(self.authServer.transport.packets, [(userauth.MSG_USERAUTH_SUCCESS, b'')])
    return d.addCallback(check)