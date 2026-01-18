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
def test_requestRaisesConchError(self):
    """
        ssh_USERAUTH_REQUEST should raise a ConchError if tryAuth returns
        None. Added to catch a bug noticed by pyflakes.
        """
    d = defer.Deferred()

    def mockCbFinishedAuth(self, ignored):
        self.fail('request should have raised ConochError')

    def mockTryAuth(kind, user, data):
        return None

    def mockEbBadAuth(reason):
        d.errback(reason.value)
    self.patch(self.authServer, 'tryAuth', mockTryAuth)
    self.patch(self.authServer, '_cbFinishedAuth', mockCbFinishedAuth)
    self.patch(self.authServer, '_ebBadAuth', mockEbBadAuth)
    packet = NS(b'user') + NS(b'none') + NS(b'public-key') + NS(b'data')
    self.authServer.ssh_USERAUTH_REQUEST(packet)
    return self.assertFailure(d, ConchError)