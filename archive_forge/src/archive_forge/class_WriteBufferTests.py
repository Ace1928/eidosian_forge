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
class WriteBufferTests(SynchronousTestCase):
    """
    Tests for L{imap4.WriteBuffer}.
    """

    def setUp(self):
        self.transport = StringTransport()

    def test_partialWrite(self):
        """
        L{imap4.WriteBuffer} buffers writes that are smaller than its
        buffer size.
        """
        buf = imap4.WriteBuffer(self.transport)
        data = b'x' * buf.bufferSize
        buf.write(data)
        self.assertFalse(self.transport.value())

    def test_overlongWrite(self):
        """
        L{imap4.WriteBuffer} writes data without buffering it when
        the size of the data exceeds the size of its buffer.
        """
        buf = imap4.WriteBuffer(self.transport)
        data = b'x' * (buf.bufferSize + 1)
        buf.write(data)
        self.assertEqual(self.transport.value(), data)

    def test_writesImplyFlush(self):
        """
        L{imap4.WriteBuffer} buffers writes until its buffer's size
        exceeds its maximum value.
        """
        buf = imap4.WriteBuffer(self.transport)
        firstData = b'x' * buf.bufferSize
        secondData = b'y'
        buf.write(firstData)
        self.assertFalse(self.transport.value())
        buf.write(secondData)
        self.assertEqual(self.transport.value(), firstData + secondData)

    def test_explicitFlush(self):
        """
        L{imap4.WriteBuffer.flush} flushes the buffer even when its
        size is smaller than the buffer size.
        """
        buf = imap4.WriteBuffer(self.transport)
        data = b'x' * buf.bufferSize
        buf.write(data)
        self.assertFalse(self.transport.value())
        buf.flush()
        self.assertEqual(self.transport.value(), data)

    def test_explicitFlushEmptyBuffer(self):
        """
        L{imap4.WriteBuffer.flush} has no effect if when the buffer is
        empty.
        """
        buf = imap4.WriteBuffer(self.transport)
        buf.flush()
        self.assertFalse(self.transport.value())