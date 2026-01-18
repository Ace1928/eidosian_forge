from __future__ import annotations
import base64
import codecs
import functools
import locale
import os
import uuid
from collections import OrderedDict
from io import BytesIO
from itertools import chain
from typing import Optional, Type
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import (
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, error, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.mail import imap4
from twisted.mail.imap4 import MessageSet
from twisted.mail.interfaces import (
from twisted.protocols import loopback
from twisted.python import failure, log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_unexpectedLoginFailure(self):
    """
        If the portal raises an exception other than
        L{UnauthorizedLogin} or L{UnhandledCredentials}, the server
        responds with a C{BAD} response and the exception is logged.
        """

    class UnexpectedException(Exception):
        """
            An unexpected exception.
            """

    class FailingChecker:
        """
            A credentials checker whose L{requestAvatarId} method
            raises L{UnexpectedException}.
            """
        credentialInterfaces = (IUsernameHashedPassword, IUsernamePassword)

        def requestAvatarId(self, credentials):
            raise UnexpectedException('Unexpected error.')
    realm = TestRealm()
    portal = Portal(realm)
    portal.registerChecker(FailingChecker())
    self.server.portal = portal
    self.server.challengers[b'LOGIN'] = loginCred = imap4.LOGINCredentials
    verifyClass(IChallengeResponse, loginCred)
    cAuth = imap4.LOGINAuthenticator(b'testuser')
    self.client.registerAuthenticator(cAuth)

    def auth():
        return self.client.authenticate(b'secret')

    def assertUnexpectedExceptionLogged():
        self.assertTrue(self.flushLoggedErrors(UnexpectedException))
    d1 = self.connected.addCallback(strip(auth))
    d1.addErrback(self.assertClientFailureMessage, b'Server error: login failed unexpectedly')
    d1.addCallback(strip(assertUnexpectedExceptionLogged))
    d1.addCallbacks(self._cbStopClient, self._ebGeneral)
    d = defer.gatherResults([self.loopback(), d1])
    return d