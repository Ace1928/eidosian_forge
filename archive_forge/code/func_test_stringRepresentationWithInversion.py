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
def test_stringRepresentationWithInversion(self):
    """
        In a L{MessageSet}, inverting the high and low numbers in a
        range doesn't affect the meaning of the range.  For example,
        3:2 displays just like 2:3, because according to the RFC they
        have the same meaning.
        """
    inputs = [imap4.parseIdList(b'2:3'), imap4.parseIdList(b'3:2')]
    outputs = ['2:3', '2:3']
    for i, o in zip(inputs, outputs):
        self.assertEqual(str(i), o)