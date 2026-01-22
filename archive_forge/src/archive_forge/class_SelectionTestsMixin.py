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
class SelectionTestsMixin(PreauthIMAP4ClientMixin):
    """
    Mixin for test cases which defines tests which apply to both I{EXAMINE} and
    I{SELECT} support.
    """

    def _examineOrSelect(self):
        """
        Issue either an I{EXAMINE} or I{SELECT} command (depending on
        C{self.method}), assert that the correct bytes are written to the
        transport, and return the L{Deferred} returned by whichever method was
        called.
        """
        d = getattr(self.client, self.method)('foobox')
        self.assertEqual(self.transport.value(), b'0001 ' + self.command + b' foobox\r\n')
        return d

    def _response(self, *lines):
        """
        Deliver the given (unterminated) response lines to C{self.client} and
        then deliver a tagged SELECT or EXAMINE completion line to finish the
        SELECT or EXAMINE response.
        """
        for line in lines:
            self.client.dataReceived(line + b'\r\n')
        self.client.dataReceived(b'0001 OK [READ-ONLY] ' + self.command + b' completed\r\n')

    def test_exists(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{EXISTS} response, the L{Deferred} return by L{IMAP4Client.select} or
        L{IMAP4Client.examine} fires with a C{dict} including the value
        associated with the C{'EXISTS'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* 3 EXISTS')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'EXISTS': 3})

    def test_nonIntegerExists(self):
        """
        If the server returns a non-integer EXISTS value in its response to a
        I{SELECT} or I{EXAMINE} command, the L{Deferred} returned by
        L{IMAP4Client.select} or L{IMAP4Client.examine} fails with
        L{IllegalServerResponse}.
        """
        d = self._examineOrSelect()
        self._response(b'* foo EXISTS')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_recent(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{RECENT} response, the L{Deferred} return by L{IMAP4Client.select} or
        L{IMAP4Client.examine} fires with a C{dict} including the value
        associated with the C{'RECENT'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* 5 RECENT')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'RECENT': 5})

    def test_nonIntegerRecent(self):
        """
        If the server returns a non-integer RECENT value in its response to a
        I{SELECT} or I{EXAMINE} command, the L{Deferred} returned by
        L{IMAP4Client.select} or L{IMAP4Client.examine} fails with
        L{IllegalServerResponse}.
        """
        d = self._examineOrSelect()
        self._response(b'* foo RECENT')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_unseen(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{UNSEEN} response, the L{Deferred} returned by L{IMAP4Client.select} or
        L{IMAP4Client.examine} fires with a C{dict} including the value
        associated with the C{'UNSEEN'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [UNSEEN 8] Message 8 is first unseen')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'UNSEEN': 8})

    def test_nonIntegerUnseen(self):
        """
        If the server returns a non-integer UNSEEN value in its response to a
        I{SELECT} or I{EXAMINE} command, the L{Deferred} returned by
        L{IMAP4Client.select} or L{IMAP4Client.examine} fails with
        L{IllegalServerResponse}.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [UNSEEN foo] Message foo is first unseen')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_uidvalidity(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{UIDVALIDITY} response, the L{Deferred} returned by
        L{IMAP4Client.select} or L{IMAP4Client.examine} fires with a C{dict}
        including the value associated with the C{'UIDVALIDITY'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [UIDVALIDITY 12345] UIDs valid')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'UIDVALIDITY': 12345})

    def test_nonIntegerUIDVALIDITY(self):
        """
        If the server returns a non-integer UIDVALIDITY value in its response to
        a I{SELECT} or I{EXAMINE} command, the L{Deferred} returned by
        L{IMAP4Client.select} or L{IMAP4Client.examine} fails with
        L{IllegalServerResponse}.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [UIDVALIDITY foo] UIDs valid')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_uidnext(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{UIDNEXT} response, the L{Deferred} returned by L{IMAP4Client.select}
        or L{IMAP4Client.examine} fires with a C{dict} including the value
        associated with the C{'UIDNEXT'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [UIDNEXT 4392] Predicted next UID')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'UIDNEXT': 4392})

    def test_nonIntegerUIDNEXT(self):
        """
        If the server returns a non-integer UIDNEXT value in its response to a
        I{SELECT} or I{EXAMINE} command, the L{Deferred} returned by
        L{IMAP4Client.select} or L{IMAP4Client.examine} fails with
        L{IllegalServerResponse}.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [UIDNEXT foo] Predicted next UID')
        self.failureResultOf(d, imap4.IllegalServerResponse)

    def test_flags(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{FLAGS} response, the L{Deferred} returned by L{IMAP4Client.select} or
        L{IMAP4Client.examine} fires with a C{dict} including the value
        associated with the C{'FLAGS'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* FLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft)')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'FLAGS': ('\\Answered', '\\Flagged', '\\Deleted', '\\Seen', '\\Draft')})

    def test_permanentflags(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{FLAGS} response, the L{Deferred} returned by L{IMAP4Client.select} or
        L{IMAP4Client.examine} fires with a C{dict} including the value
        associated with the C{'FLAGS'} key.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [PERMANENTFLAGS (\\Starred)] Just one permanent flag in that list up there')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False, 'PERMANENTFLAGS': ('\\Starred',)})

    def test_unrecognizedOk(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{OK} with unrecognized response code text, parsing does not fail.
        """
        d = self._examineOrSelect()
        self._response(b'* OK [X-MADE-UP] I just made this response text up.')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False})

    def test_bareOk(self):
        """
        If the server response to a I{SELECT} or I{EXAMINE} command includes an
        I{OK} with no response code text, parsing does not fail.
        """
        d = self._examineOrSelect()
        self._response(b'* OK')
        self.assertEqual(self.successResultOf(d), {'READ-WRITE': False})