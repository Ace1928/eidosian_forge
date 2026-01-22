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
class SendmailTests(TestCase):
    """
    Tests for L{twisted.mail.smtp.sendmail}.
    """

    def test_defaultReactorIsGlobalReactor(self):
        """
        The default C{reactor} parameter of L{twisted.mail.smtp.sendmail} is
        L{twisted.internet.reactor}.
        """
        fullSpec = inspect.getfullargspec(smtp.sendmail)
        defaults = fullSpec[3]
        self.assertEqual(reactor, defaults[2])

    def _honorsESMTPArguments(self, username, password):
        """
        L{twisted.mail.smtp.sendmail} creates the ESMTP factory with the ESMTP
        arguments.
        """
        reactor = MemoryReactor()
        smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor, username=username, password=password, requireTransportSecurity=True, requireAuthentication=True)
        factory = reactor.tcpClients[0][2]
        self.assertEqual(factory._requireTransportSecurity, True)
        self.assertEqual(factory._requireAuthentication, True)
        self.assertEqual(factory.username, b'foo')
        self.assertEqual(factory.password, b'bar')

    def test_honorsESMTPArgumentsUnicodeUserPW(self):
        """
        L{twisted.mail.smtp.sendmail} should accept C{username} and C{password}
        which are L{unicode}.
        """
        return self._honorsESMTPArguments(username='foo', password='bar')

    def test_honorsESMTPArgumentsBytesUserPW(self):
        """
        L{twisted.mail.smtp.sendmail} should accept C{username} and C{password}
        which are L{bytes}.
        """
        return self._honorsESMTPArguments(username=b'foo', password=b'bar')

    def test_messageFilePassthrough(self):
        """
        L{twisted.mail.smtp.sendmail} will pass through the message untouched
        if it is a file-like object.
        """
        reactor = MemoryReactor()
        messageFile = BytesIO(b'File!')
        smtp.sendmail('localhost', 'source@address', 'recipient@address', messageFile, reactor=reactor)
        factory = reactor.tcpClients[0][2]
        self.assertIs(factory.file, messageFile)

    def test_messageStringMadeFile(self):
        """
        L{twisted.mail.smtp.sendmail} will turn non-file-like objects
        (eg. strings) into file-like objects before sending.
        """
        reactor = MemoryReactor()
        smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor)
        factory = reactor.tcpClients[0][2]
        messageFile = factory.file
        messageFile.seek(0)
        self.assertEqual(messageFile.read(), b'message')

    def test_senderDomainName(self):
        """
        L{twisted.mail.smtp.sendmail} passes through the sender domain name, if
        provided.
        """
        reactor = MemoryReactor()
        smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor, senderDomainName='foo')
        factory = reactor.tcpClients[0][2]
        self.assertEqual(factory.domain, b'foo')

    def test_cancelBeforeConnectionMade(self):
        """
        When a user cancels L{twisted.mail.smtp.sendmail} before the connection
        is made, the connection is closed by
        L{twisted.internet.interfaces.IConnector.disconnect}.
        """
        reactor = MemoryReactor()
        d = smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor)
        d.cancel()
        self.assertEqual(reactor.connectors[0]._disconnected, True)
        failure = self.failureResultOf(d)
        failure.trap(defer.CancelledError)

    def test_cancelAfterConnectionMade(self):
        """
        When a user cancels L{twisted.mail.smtp.sendmail} after the connection
        is made, the connection is closed by
        L{twisted.internet.interfaces.ITransport.abortConnection}.
        """
        reactor = MemoryReactor()
        transport = AbortableStringTransport()
        d = smtp.sendmail('localhost', 'source@address', 'recipient@address', b'message', reactor=reactor)
        factory = reactor.tcpClients[0][2]
        p = factory.buildProtocol(None)
        p.makeConnection(transport)
        d.cancel()
        self.assertEqual(transport.aborting, True)
        self.assertEqual(transport.disconnecting, True)
        failure = self.failureResultOf(d)
        failure.trap(defer.CancelledError)