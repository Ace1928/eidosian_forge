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
class SMTPServerTests(TestCase):
    """
    Test various behaviors of L{twisted.mail.smtp.SMTP} and
    L{twisted.mail.smtp.ESMTP}.
    """

    def testSMTPGreetingHost(self, serverClass=smtp.SMTP):
        """
        Test that the specified hostname shows up in the SMTP server's
        greeting.
        """
        s = serverClass()
        s.host = b'example.com'
        t = StringTransport()
        s.makeConnection(t)
        s.connectionLost(error.ConnectionDone())
        self.assertIn(b'example.com', t.value())

    def testSMTPGreetingNotExtended(self):
        """
        Test that the string "ESMTP" does not appear in the SMTP server's
        greeting since that string strongly suggests the presence of support
        for various SMTP extensions which are not supported by L{smtp.SMTP}.
        """
        s = smtp.SMTP()
        t = StringTransport()
        s.makeConnection(t)
        s.connectionLost(error.ConnectionDone())
        self.assertNotIn(b'ESMTP', t.value())

    def testESMTPGreetingHost(self):
        """
        Similar to testSMTPGreetingHost, but for the L{smtp.ESMTP} class.
        """
        self.testSMTPGreetingHost(smtp.ESMTP)

    def testESMTPGreetingExtended(self):
        """
        Test that the string "ESMTP" does appear in the ESMTP server's
        greeting since L{smtp.ESMTP} does support the SMTP extensions which
        that advertises to the client.
        """
        s = smtp.ESMTP()
        t = StringTransport()
        s.makeConnection(t)
        s.connectionLost(error.ConnectionDone())
        self.assertIn(b'ESMTP', t.value())

    def test_SMTPUnknownCommand(self):
        """
        Sending an unimplemented command is responded to with a 500.
        """
        s = smtp.SMTP()
        t = StringTransport()
        s.makeConnection(t)
        s.lineReceived(b'DOAGOODTHING')
        s.connectionLost(error.ConnectionDone())
        self.assertIn(b'500 Command not implemented', t.value())

    def test_acceptSenderAddress(self):
        """
        Test that a C{MAIL FROM} command with an acceptable address is
        responded to with the correct success code.
        """

        class AcceptanceDelivery(NotImplementedDelivery):
            """
            Delivery object which accepts all senders as valid.
            """

            def validateFrom(self, helo, origin):
                return origin
        realm = SingletonRealm(smtp.IMessageDelivery, AcceptanceDelivery())
        portal = Portal(realm, [AllowAnonymousAccess()])
        proto = smtp.SMTP()
        proto.portal = portal
        trans = StringTransport()
        proto.makeConnection(trans)
        proto.dataReceived(b'HELO example.com\r\n')
        trans.clear()
        proto.dataReceived(b'MAIL FROM:<alice@example.com>\r\n')
        proto.connectionLost(error.ConnectionLost())
        self.assertEqual(trans.value(), b'250 Sender address accepted\r\n')

    def test_deliveryRejectedSenderAddress(self):
        """
        Test that a C{MAIL FROM} command with an address rejected by a
        L{smtp.IMessageDelivery} instance is responded to with the correct
        error code.
        """

        class RejectionDelivery(NotImplementedDelivery):
            """
            Delivery object which rejects all senders as invalid.
            """

            def validateFrom(self, helo, origin):
                raise smtp.SMTPBadSender(origin)
        realm = SingletonRealm(smtp.IMessageDelivery, RejectionDelivery())
        portal = Portal(realm, [AllowAnonymousAccess()])
        proto = smtp.SMTP()
        proto.portal = portal
        trans = StringTransport()
        proto.makeConnection(trans)
        proto.dataReceived(b'HELO example.com\r\n')
        trans.clear()
        proto.dataReceived(b'MAIL FROM:<alice@example.com>\r\n')
        proto.connectionLost(error.ConnectionLost())
        self.assertEqual(trans.value(), b'550 Cannot receive from specified address <alice@example.com>: Sender not acceptable\r\n')

    @implementer(ICredentialsChecker)
    def test_portalRejectedSenderAddress(self):
        """
        Test that a C{MAIL FROM} command with an address rejected by an
        L{smtp.SMTP} instance's portal is responded to with the correct error
        code.
        """

        class DisallowAnonymousAccess:
            """
            Checker for L{IAnonymous} which rejects authentication attempts.
            """
            credentialInterfaces = (IAnonymous,)

            def requestAvatarId(self, credentials):
                return defer.fail(UnauthorizedLogin())
        realm = SingletonRealm(smtp.IMessageDelivery, NotImplementedDelivery())
        portal = Portal(realm, [DisallowAnonymousAccess()])
        proto = smtp.SMTP()
        proto.portal = portal
        trans = StringTransport()
        proto.makeConnection(trans)
        proto.dataReceived(b'HELO example.com\r\n')
        trans.clear()
        proto.dataReceived(b'MAIL FROM:<alice@example.com>\r\n')
        proto.connectionLost(error.ConnectionLost())
        self.assertEqual(trans.value(), b'550 Cannot receive from specified address <alice@example.com>: Sender not acceptable\r\n')

    def test_portalRejectedAnonymousSender(self):
        """
        Test that a C{MAIL FROM} command issued without first authenticating
        when a portal has been configured to disallow anonymous logins is
        responded to with the correct error code.
        """
        realm = SingletonRealm(smtp.IMessageDelivery, NotImplementedDelivery())
        portal = Portal(realm, [])
        proto = smtp.SMTP()
        proto.portal = portal
        trans = StringTransport()
        proto.makeConnection(trans)
        proto.dataReceived(b'HELO example.com\r\n')
        trans.clear()
        proto.dataReceived(b'MAIL FROM:<alice@example.com>\r\n')
        proto.connectionLost(error.ConnectionLost())
        self.assertEqual(trans.value(), b'550 Cannot receive from specified address <alice@example.com>: Unauthenticated senders not allowed\r\n')