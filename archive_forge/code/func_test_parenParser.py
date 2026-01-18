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
def test_parenParser(self):
    s = b'\r\n'.join([b'xx'] * 4)

    def check(case, expected):
        parsed = imap4.parseNestedParens(case)
        self.assertEqual(parsed, [expected])
    check(b'(BODY.PEEK[HEADER.FIELDS.NOT (subject bcc cc)] {%d}\r\n%b)' % (len(s), s), [b'BODY.PEEK', [b'HEADER.FIELDS.NOT', [b'subject', b'bcc', b'cc']], s])
    check(b'(FLAGS (\\Seen) INTERNALDATE "17-Jul-1996 02:44:25 -0700" RFC822.SIZE 4286 ENVELOPE ("Wed, 17 Jul 1996 02:23:25 -0700 (PDT)" "IMAP4rev1 WG mtg summary and minutes" (("Terry Gray" NIL gray cac.washington.edu)) (("Terry Gray" NIL gray cac.washington.edu)) (("Terry Gray" NIL gray cac.washington.edu)) ((NIL NIL imap cac.washington.edu)) ((NIL NIL minutes CNRI.Reston.VA.US) ("John Klensin" NIL KLENSIN INFOODS.MIT.EDU)) NIL NIL <B27397-0100000@cac.washington.edu>) BODY (TEXT PLAIN (CHARSET US-ASCII) NIL NIL 7BIT 3028 92))', [b'FLAGS', [b'\\Seen'], b'INTERNALDATE', b'17-Jul-1996 02:44:25 -0700', b'RFC822.SIZE', b'4286', b'ENVELOPE', [b'Wed, 17 Jul 1996 02:23:25 -0700 (PDT)', b'IMAP4rev1 WG mtg summary and minutes', [[b'Terry Gray', None, b'gray', b'cac.washington.edu']], [[b'Terry Gray', None, b'gray', b'cac.washington.edu']], [[b'Terry Gray', None, b'gray', b'cac.washington.edu']], [[None, None, b'imap', b'cac.washington.edu']], [[None, None, b'minutes', b'CNRI.Reston.VA.US'], [b'John Klensin', None, b'KLENSIN', b'INFOODS.MIT.EDU']], None, None, b'<B27397-0100000@cac.washington.edu>'], b'BODY', [b'TEXT', b'PLAIN', [b'CHARSET', b'US-ASCII'], None, None, b'7BIT', b'3028', b'92']])
    check(b'("oo \\"oo\\" oo")', [b'oo "oo" oo'])
    check(b'("oo \\\\ oo")', [b'oo \\\\ oo'])
    check(b'("oo \\ oo")', [b'oo \\ oo'])
    check(b'("oo \\o")', [b'oo \\o'])
    check(b'("oo \\o")', [b'oo \\o'])
    check(b'(oo \\o)', [b'oo', b'\\o'])
    check(b'(oo \\o)', [b'oo', b'\\o'])