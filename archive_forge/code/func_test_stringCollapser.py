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
def test_stringCollapser(self):
    cases = [[b'a', b'b', b'c', b'd', b'e'], [b'a', b' ', b'"', b'b', b'c', b' ', b'"', b' ', b'd', b'e'], [[b'a', b'b', b'c'], b'd', b'e'], [b'a', [b'b', b'c', b'd'], b'e'], [b'a', b'b', [b'c', b'd', b'e']], [b'"', b'a', b' ', b'"', [b'b', b'c', b'd'], b'"', b' ', b'e', b'"'], [b'a', [b'"', b' ', b'b', b'c', b' ', b' ', b'"'], b'd', b'e']]
    answers = [[b'abcde'], [b'a', b'bc ', b'de'], [[b'abc'], b'de'], [b'a', [b'bcd'], b'e'], [b'ab', [b'cde']], [b'a ', [b'bcd'], b' e'], [b'a', [b' bc  '], b'de']]
    for case, expected in zip(cases, answers):
        self.assertEqual(imap4.collapseStrings(case), expected)