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
class LiteralFileTests(LiteralTestsMixin, TestCase):
    """
    Tests for L{imap4.LiteralFile}.
    """
    literalFactory = imap4.LiteralFile

    def test_callback(self):
        """
        Calling L{imap4.LiteralFile.callback} with a line fires the
        instance's L{Deferred} with a 2-L{tuple} whose first element
        is the file and whose second is the provided line.
        """
        data = b'data'
        extra = b'extra'
        literal = imap4.LiteralFile(len(data), self.deferred)
        for c in iterbytes(data):
            literal.write(c)
        literal.callback(b'extra')
        result = self.successResultOf(self.deferred)
        self.assertEqual(len(result), 2)
        dataFile, extra = result
        self.assertEqual(dataFile.read(), b'data')

    def test_callbackSpooledToDisk(self):
        """
        A L{imap4.LiteralFile} whose size exceeds the maximum
        in-memory size spools its content to disk, and invoking its
        L{callback} with a line fires the instance's L{Deferred} with
        a 2-L{tuple} whose first element is the spooled file and whose second
        is the provided line.
        """
        data = b'data'
        extra = b'extra'
        self.patch(imap4.LiteralFile, '_memoryFileLimit', 1)
        literal = imap4.LiteralFile(len(data), self.deferred)
        for c in iterbytes(data):
            literal.write(c)
        literal.callback(b'extra')
        result = self.successResultOf(self.deferred)
        self.assertEqual(len(result), 2)
        dataFile, extra = result
        self.assertEqual(dataFile.read(), b'data')