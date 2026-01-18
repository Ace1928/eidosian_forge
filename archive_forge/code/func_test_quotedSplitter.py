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
def test_quotedSplitter(self):
    cases = [b'Hello World', b'Hello "World!"', b'World "Hello" "How are you?"', b'"Hello world" How "are you?"', b'foo bar "baz buz" NIL', b'foo bar "baz buz" "NIL"', b'foo NIL "baz buz" bar', b'foo "NIL" "baz buz" bar', b'"NIL" bar "baz buz" foo', b'oo \\"oo\\" oo', b'"oo \\"oo\\" oo"', b'oo \t oo', b'"oo \t oo"', b'oo \\t oo', b'"oo \\t oo"', b'oo \\o oo', b'"oo \\o oo"', b'oo \\o oo', b'"oo \\o oo"']
    answers = [[b'Hello', b'World'], [b'Hello', b'World!'], [b'World', b'Hello', b'How are you?'], [b'Hello world', b'How', b'are you?'], [b'foo', b'bar', b'baz buz', None], [b'foo', b'bar', b'baz buz', b'NIL'], [b'foo', None, b'baz buz', b'bar'], [b'foo', b'NIL', b'baz buz', b'bar'], [b'NIL', b'bar', b'baz buz', b'foo'], [b'oo', b'"oo"', b'oo'], [b'oo "oo" oo'], [b'oo', b'oo'], [b'oo \t oo'], [b'oo', b'\\t', b'oo'], [b'oo \\t oo'], [b'oo', b'\\o', b'oo'], [b'oo \\o oo'], [b'oo', b'\\o', b'oo'], [b'oo \\o oo']]
    errors = [b'"mismatched quote', b'mismatched quote"', b'mismatched"quote', b'"oops here is" another"']
    for s in errors:
        self.assertRaises(imap4.MismatchedQuoting, imap4.splitQuoted, s)
    for case, expected in zip(cases, answers):
        self.assertEqual(imap4.splitQuoted(case), expected)