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
class UnsolicitedResponseTests(IMAP4HelperMixin, TestCase):

    def testReadWrite(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def loggedIn():
            self.server.modeChanged(1)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(loggedIn)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestReadWrite)

    def _cbTestReadWrite(self, ignored):
        E = self.client.events
        self.assertEqual(E, [['modeChanged', 1]])

    def testReadOnly(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def loggedIn():
            self.server.modeChanged(0)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(loggedIn)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestReadOnly)

    def _cbTestReadOnly(self, ignored):
        E = self.client.events
        self.assertEqual(E, [['modeChanged', 0]])

    def testFlagChange(self):
        flags = {1: ['\\Answered', '\\Deleted'], 5: [], 10: ['\\Recent']}

        def login():
            return self.client.login(b'testuser', b'password-test')

        def loggedIn():
            self.server.flagsChanged(flags)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(loggedIn)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestFlagChange, flags)

    def _cbTestFlagChange(self, ignored, flags):
        E = self.client.events
        expect = [['flagsChanged', {x[0]: x[1]}] for x in flags.items()]
        E.sort(key=lambda o: o[0])
        expect.sort(key=lambda o: o[0])
        self.assertEqual(E, expect)

    def testNewMessages(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def loggedIn():
            self.server.newMessages(10, None)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(loggedIn)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestNewMessages)

    def _cbTestNewMessages(self, ignored):
        E = self.client.events
        self.assertEqual(E, [['newMessages', 10, None]])

    def testNewRecentMessages(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def loggedIn():
            self.server.newMessages(None, 10)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(loggedIn)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestNewRecentMessages)

    def _cbTestNewRecentMessages(self, ignored):
        E = self.client.events
        self.assertEqual(E, [['newMessages', None, 10]])

    def testNewMessagesAndRecent(self):

        def login():
            return self.client.login(b'testuser', b'password-test')

        def loggedIn():
            self.server.newMessages(20, 10)
        d1 = self.connected.addCallback(strip(login))
        d1.addCallback(strip(loggedIn)).addErrback(self._ebGeneral)
        d = defer.gatherResults([self.loopback(), d1])
        return d.addCallback(self._cbTestNewMessagesAndRecent)

    def _cbTestNewMessagesAndRecent(self, ignored):
        E = self.client.events
        self.assertEqual(E, [['newMessages', 20, None], ['newMessages', None, 10]])