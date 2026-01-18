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
def test_noneAuthentication(self):
    """
        A client may request a list of authentication 'method name' values
        that may continue by using the "none" authentication 'method name'.

        See RFC 4252 Section 5.2.
        """
    d = self.authServer.ssh_USERAUTH_REQUEST(NS(b'foo') + NS(b'service') + NS(b'none'))
    return d.addCallback(self._checkFailed)