import base64
import inspect
import re
from io import BytesIO
from typing import Any, List, Optional, Tuple, Type
from zope.interface import directlyProvides, implementer
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.error
import twisted.cred.portal
from twisted import cred
from twisted.cred.checkers import AllowAnonymousAccess, ICredentialsChecker
from twisted.cred.credentials import IAnonymous
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import address, defer, error, interfaces, protocol, reactor, task
from twisted.internet.testing import MemoryReactor, StringTransport
from twisted.mail import smtp
from twisted.mail._cred import LOGINCredentials
from twisted.protocols import basic, loopback
from twisted.python.util import LineLog
from twisted.trial.unittest import TestCase
def assertServerAuthenticated(self, loginArgs, username=b'username', password=b'password'):
    """
        Assert that a login attempt has been made, that the credentials and
        interfaces passed to it are correct, and that when the login request
        is satisfied, a successful response is sent by the ESMTP server
        instance.

        @param loginArgs: A C{list} previously passed to L{portalFactory}.
        @param username: The login user.
        @param password: The login password.
        """
    d, credentials, mind, interfaces = loginArgs.pop()
    self.assertEqual(loginArgs, [])
    self.assertTrue(twisted.cred.credentials.IUsernamePassword.providedBy(credentials))
    self.assertEqual(credentials.username, username)
    self.assertTrue(credentials.checkPassword(password))
    self.assertIn(smtp.IMessageDeliveryFactory, interfaces)
    self.assertIn(smtp.IMessageDelivery, interfaces)
    d.callback((smtp.IMessageDeliveryFactory, None, lambda: None))
    self.assertEqual([b'235 Authentication successful.'], self.transport.value().splitlines())