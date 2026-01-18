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
def test_resetTimeoutWhileSending(self):
    """
        The timeout is not allowed to expire after the server has accepted a
        DATA command and the client is actively sending data to it.
        """

    class SlowFile:
        """
            A file-like which returns one byte from each read call until the
            specified number of bytes have been returned.
            """

        def __init__(self, size):
            self._size = size

        def read(self, max=None):
            if self._size:
                self._size -= 1
                return b'x'
            return b''
    failed = []
    onDone = defer.Deferred()
    onDone.addErrback(failed.append)
    clientFactory = smtp.SMTPSenderFactory('source@address', 'recipient@address', SlowFile(1), onDone, retries=0, timeout=3)
    clientFactory.domain = b'example.org'
    clock = task.Clock()
    client = clientFactory.buildProtocol(address.IPv4Address('TCP', 'example.net', 25))
    client.callLater = clock.callLater
    transport = StringTransport()
    client.makeConnection(transport)
    client.dataReceived(b'220 Ok\r\n250 Ok\r\n250 Ok\r\n250 Ok\r\n354 Ok\r\n')
    self.assertNotIdentical(transport.producer, None)
    self.assertFalse(transport.streaming)
    clock.advance(2)
    self.assertEqual(failed, [])
    transport.producer.resumeProducing()
    clock.advance(2)
    self.assertEqual(failed, [])
    transport.producer.resumeProducing()
    client.dataReceived(b'250 Ok\r\n')
    self.assertEqual(failed, [])
    self.assertEqual(transport.value(), b'HELO example.org\r\nMAIL FROM:<source@address>\r\nRCPT TO:<recipient@address>\r\nDATA\r\nx\r\n.\r\nRSET\r\n')