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
@skipIf(not ClientTLSContext, 'OpenSSL not present')
@skipIf(not interfaces.IReactorSSL(reactor, None), "Reactor doesn't support SSL")
class TLSTests(IMAP4HelperMixin, TestCase):
    serverCTX = None
    clientCTX = None
    if ServerTLSContext:
        serverCTX = ServerTLSContext()
    if ClientTLSContext:
        clientCTX = ClientTLSContext()

    def loopback(self):
        return loopback.loopbackTCP(self.server, self.client, noisy=False)

    def testAPileOfThings(self):
        SimpleServer.theAccount.addMailbox(b'inbox')
        called = []

        def login():
            called.append(None)
            return self.client.login(b'testuser', b'password-test')

        def list():
            called.append(None)
            return self.client.list(b'inbox', b'%')

        def status():
            called.append(None)
            return self.client.status(b'inbox', 'UIDNEXT')

        def examine():
            called.append(None)
            return self.client.examine(b'inbox')

        def logout():
            called.append(None)
            return self.client.logout()
        self.client.requireTransportSecurity = True
        methods = [login, list, status, examine, logout]
        for method in methods:
            self.connected.addCallback(strip(method))
        self.connected.addCallbacks(self._cbStopClient, self._ebGeneral)

        def check(ignored):
            self.assertEqual(self.server.startedTLS, True)
            self.assertEqual(self.client.startedTLS, True)
            self.assertEqual(len(called), len(methods))
        d = self.loopback()
        d.addCallback(check)
        return d

    def testLoginLogin(self):
        self.server.checker.addUser(b'testuser', b'password-test')
        success = []
        self.client.registerAuthenticator(imap4.LOGINAuthenticator(b'testuser'))
        self.connected.addCallback(lambda _: self.client.authenticate(b'password-test')).addCallback(lambda _: self.client.logout()).addCallback(success.append).addCallback(self._cbStopClient).addErrback(self._ebGeneral)
        d = self.loopback()
        d.addCallback(lambda x: self.assertEqual(len(success), 1))
        return d

    def startTLSAndAssertSession(self):
        """
        Begin a C{STARTTLS} sequence and assert that it results in a
        TLS session.

        @return: A L{Deferred} that fires when the underlying
            connection between the client and server has been terminated.
        """
        success = []
        self.connected.addCallback(strip(self.client.startTLS))

        def checkSecure(ignored):
            self.assertTrue(interfaces.ISSLTransport.providedBy(self.client.transport))
        self.connected.addCallback(checkSecure)
        self.connected.addCallback(success.append)
        d = self.loopback()
        d.addCallback(lambda x: self.assertTrue(success))
        return defer.gatherResults([d, self.connected])

    def test_startTLS(self):
        """
        L{IMAP4Client.startTLS} triggers TLS negotiation and returns a
        L{Deferred} which fires after the client's transport is using
        encryption.
        """
        disconnected = self.startTLSAndAssertSession()
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return disconnected

    def test_startTLSDefault(self) -> Deferred[object]:
        """
        L{IMAPClient.startTLS} supplies a default TLS context if none is
        supplied.
        """
        self.assertIsNotNone(self.client.context)
        self.client.context = None
        disconnected: Deferred[object] = self.startTLSAndAssertSession()
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return disconnected

    def test_doubleSTARTTLS(self):
        """
        A server that receives a second C{STARTTLS} sends a C{NO}
        response.
        """

        class DoubleSTARTTLSClient(SimpleClient):

            def startTLS(self):
                if not self.startedTLS:
                    return SimpleClient.startTLS(self)
                return self.sendCommand(imap4.Command(b'STARTTLS'))
        self.client = DoubleSTARTTLSClient(self.connected, contextFactory=self.clientCTX)
        disconnected = self.startTLSAndAssertSession()
        self.connected.addCallback(strip(self.client.startTLS))
        self.connected.addErrback(self.assertClientFailureMessage, b'TLS already negotiated')
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return disconnected

    def test_startTLSWithExistingChallengers(self):
        """
        Starting a TLS negotiation with an L{IMAP4Server} that already
        has C{LOGIN} and C{PLAIN} L{IChallengeResponse} factories uses
        those factories.
        """
        self.server.challengers = {b'LOGIN': imap4.LOGINCredentials, b'PLAIN': imap4.PLAINCredentials}

        @defer.inlineCallbacks
        def assertLOGINandPLAIN():
            capabilities = (yield self.client.getCapabilities())
            self.assertIn(b'AUTH', capabilities)
            self.assertIn(b'LOGIN', capabilities[b'AUTH'])
            self.assertIn(b'PLAIN', capabilities[b'AUTH'])
        self.connected.addCallback(strip(assertLOGINandPLAIN))
        disconnected = self.startTLSAndAssertSession()
        self.connected.addCallback(strip(assertLOGINandPLAIN))
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return disconnected

    def test_loginBeforeSTARTTLS(self):
        """
        A client that attempts to log in before issuing the
        C{STARTTLS} command receives a C{NO} response.
        """
        self.client.startTLS = lambda: defer.succeed(([], 'OK Begin TLS negotiation now'))
        self.connected.addCallback(lambda _: self.client.login(b'wrong', b'time'))
        self.connected.addErrback(self.assertClientFailureMessage, b'LOGIN is disabled before STARTTLS')
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return defer.gatherResults([self.loopback(), self.connected])

    def testFailedStartTLS(self):
        failures = []

        def breakServerTLS(ign):
            self.server.canStartTLS = False
        self.connected.addCallback(breakServerTLS)
        self.connected.addCallback(lambda ign: self.client.startTLS())
        self.connected.addErrback(lambda err: failures.append(err.trap(imap4.IMAP4Exception)))
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)

        def check(ignored):
            self.assertTrue(failures)
            self.assertIdentical(failures[0], imap4.IMAP4Exception)
        return self.loopback().addCallback(check)