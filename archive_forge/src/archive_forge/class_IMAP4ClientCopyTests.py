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
class IMAP4ClientCopyTests(PreauthIMAP4ClientMixin, SynchronousTestCase):
    """
    Tests for the L{IMAP4Client.copy} method.

    An example of the C{COPY} command, which this method implements,
    from RFC 3501, section 6.4.7::

        C: A003 COPY 2:4 MEETING
        S: A003 OK COPY completed
    """
    clientProtocol = StillSimplerClient

    def test_copySequenceNumbers(self):
        """
        L{IMAP4Client.copy} copies the messages identified by their
        sequence numbers to the mailbox, returning a L{Deferred} that
        succeeds with a true value.
        """
        d = self.client.copy('2:3', 'MEETING', uid=False)
        self.assertEqual(self.transport.value(), b'0001 COPY 2:3 MEETING\r\n')
        self.client.lineReceived(b'0001 OK COPY completed')
        self.assertEqual(self.successResultOf(d), ([], b'OK COPY completed'))

    def test_copySequenceNumbersFails(self):
        """
        L{IMAP4Client.copy} returns a L{Deferred} that fails with an
        L{IMAP4Exception} when the messages specified by the given
        sequence numbers could not be copied to the mailbox.
        """
        d = self.client.copy('2:3', 'MEETING', uid=False)
        self.assertEqual(self.transport.value(), b'0001 COPY 2:3 MEETING\r\n')
        self.client.lineReceived(b'0001 BAD COPY failed')
        self.assertIsInstance(self.failureResultOf(d).value, imap4.IMAP4Exception)

    def test_copyUIDs(self):
        """
        L{IMAP4Client.copy} copies the messages identified by their
        UIDs to the mailbox, returning a L{Deferred} that succeeds
        with a true value.
        """
        d = self.client.copy('2:3', 'MEETING', uid=True)
        self.assertEqual(self.transport.value(), b'0001 UID COPY 2:3 MEETING\r\n')
        self.client.lineReceived(b'0001 OK COPY completed')
        self.assertEqual(self.successResultOf(d), ([], b'OK COPY completed'))

    def test_copyUIDsFails(self):
        """
        L{IMAP4Client.copy} returns a L{Deferred} that fails with an
        L{IMAP4Exception} when the messages specified by the given
        UIDs could not be copied to the mailbox.
        """
        d = self.client.copy('2:3', 'MEETING', uid=True)
        self.assertEqual(self.transport.value(), b'0001 UID COPY 2:3 MEETING\r\n')
        self.client.lineReceived(b'0001 BAD COPY failed')
        self.assertIsInstance(self.failureResultOf(d).value, imap4.IMAP4Exception)