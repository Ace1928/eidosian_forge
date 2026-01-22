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
def _timeoutTest(self, onDone, clientFactory):
    """
        Connect the clientFactory, and check the timeout on the request.
        """
    clock = task.Clock()
    client = clientFactory.buildProtocol(address.IPv4Address('TCP', 'example.net', 25))
    client.callLater = clock.callLater
    t = StringTransport()
    client.makeConnection(t)
    t.protocol = client

    def check(ign):
        self.assertEqual(clock.seconds(), 0.5)
    d = self.assertFailure(onDone, smtp.SMTPTimeoutError).addCallback(check)
    clock.advance(0.1)
    clock.advance(0.4)
    return d