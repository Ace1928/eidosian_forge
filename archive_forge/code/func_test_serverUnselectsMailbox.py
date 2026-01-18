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
def test_serverUnselectsMailbox(self):
    """
        The server unsets the selected mailbox when timing out a
        connection.
        """
    self.patch(SimpleServer.theAccount, 'mailboxFactory', UncloseableMailbox)
    SimpleServer.theAccount.addMailbox('mailbox-test')
    mbox = SimpleServer.theAccount.mailboxes['MAILBOX-TEST']
    self.assertFalse(ICloseableMailboxIMAP.providedBy(mbox))
    c = Clock()
    self.server.callLater = c.callLater

    def login():
        return self.client.login(b'testuser', b'password-test')

    def select():
        return self.client.select('mailbox-test')

    def assertSet():
        self.assertIs(mbox, self.server.mbox)

    def expireTime():
        c.advance(self.server.POSTAUTH_TIMEOUT * 2)

    def assertUnset():
        self.assertFalse(self.server.mbox)
    d = self.connected.addCallback(strip(login))
    d.addCallback(strip(select))
    d.addCallback(strip(assertSet))
    d.addCallback(strip(expireTime))
    d.addCallback(strip(assertUnset))
    return defer.gatherResults([d, self.loopback()])