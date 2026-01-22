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
class MessageProducerTests(SynchronousTestCase):

    def testSinglePart(self):
        body = b'This is body text.  Rar.'
        headers = OrderedDict()
        headers['from'] = 'sender@host'
        headers['to'] = 'recipient@domain'
        headers['subject'] = 'booga booga boo'
        headers['content-type'] = 'text/plain'
        msg = FakeyMessage(headers, (), None, body, 123, None)
        c = BufferingConsumer()
        p = imap4.MessageProducer(msg)
        d = p.beginProducing(c)

        def cbProduced(result):
            self.assertIdentical(result, p)
            self.assertEqual(b''.join(c.buffer), b'{119}\r\nFrom: sender@host\r\nTo: recipient@domain\r\nSubject: booga booga boo\r\nContent-Type: text/plain\r\n\r\n' + body)
        return d.addCallback(cbProduced)

    def testSingleMultiPart(self):
        outerBody = b''
        innerBody = b'Contained body message text.  Squarge.'
        headers = OrderedDict()
        headers['from'] = 'sender@host'
        headers['to'] = 'recipient@domain'
        headers['subject'] = 'booga booga boo'
        headers['content-type'] = 'multipart/alternative; boundary="xyz"'
        innerHeaders = OrderedDict()
        innerHeaders['subject'] = 'this is subject text'
        innerHeaders['content-type'] = 'text/plain'
        msg = FakeyMessage(headers, (), None, outerBody, 123, [FakeyMessage(innerHeaders, (), None, innerBody, None, None)])
        c = BufferingConsumer()
        p = imap4.MessageProducer(msg)
        d = p.beginProducing(c)

        def cbProduced(result):
            self.failUnlessIdentical(result, p)
            self.assertEqual(b''.join(c.buffer), b'{239}\r\nFrom: sender@host\r\nTo: recipient@domain\r\nSubject: booga booga boo\r\nContent-Type: multipart/alternative; boundary="xyz"\r\n\r\n\r\n--xyz\r\nSubject: this is subject text\r\nContent-Type: text/plain\r\n\r\n' + innerBody + b'\r\n--xyz--\r\n')
        return d.addCallback(cbProduced)

    def testMultipleMultiPart(self):
        outerBody = b''
        innerBody1 = b'Contained body message text.  Squarge.'
        innerBody2 = b'Secondary <i>message</i> text of squarge body.'
        headers = OrderedDict()
        headers['from'] = 'sender@host'
        headers['to'] = 'recipient@domain'
        headers['subject'] = 'booga booga boo'
        headers['content-type'] = 'multipart/alternative; boundary="xyz"'
        innerHeaders = OrderedDict()
        innerHeaders['subject'] = 'this is subject text'
        innerHeaders['content-type'] = 'text/plain'
        innerHeaders2 = OrderedDict()
        innerHeaders2['subject'] = '<b>this is subject</b>'
        innerHeaders2['content-type'] = 'text/html'
        msg = FakeyMessage(headers, (), None, outerBody, 123, [FakeyMessage(innerHeaders, (), None, innerBody1, None, None), FakeyMessage(innerHeaders2, (), None, innerBody2, None, None)])
        c = BufferingConsumer()
        p = imap4.MessageProducer(msg)
        d = p.beginProducing(c)

        def cbProduced(result):
            self.failUnlessIdentical(result, p)
            self.assertEqual(b''.join(c.buffer), b'{354}\r\nFrom: sender@host\r\nTo: recipient@domain\r\nSubject: booga booga boo\r\nContent-Type: multipart/alternative; boundary="xyz"\r\n\r\n\r\n--xyz\r\nSubject: this is subject text\r\nContent-Type: text/plain\r\n\r\n' + innerBody1 + b'\r\n--xyz\r\nSubject: <b>this is subject</b>\r\nContent-Type: text/html\r\n\r\n' + innerBody2 + b'\r\n--xyz--\r\n')
        return d.addCallback(cbProduced)

    def test_multiPartNoBoundary(self):
        """
        A boundary is generated if none is provided.
        """
        outerBody = b''
        innerBody = b'Contained body message text.  Squarge.'
        headers = OrderedDict()
        headers['from'] = 'sender@host'
        headers['to'] = 'recipient@domain'
        headers['subject'] = 'booga booga boo'
        headers['content-type'] = 'multipart/alternative'
        innerHeaders = OrderedDict()
        innerHeaders['subject'] = 'this is subject text'
        innerHeaders['content-type'] = 'text/plain'
        msg = FakeyMessage(headers, (), None, outerBody, 123, [FakeyMessage(innerHeaders, (), None, innerBody, None, None)])
        c = BufferingConsumer()
        p = imap4.MessageProducer(msg)
        p._uuid4 = lambda: uuid.UUID('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa')
        d = p.beginProducing(c)

        def cbProduced(result):
            self.failUnlessIdentical(result, p)
            self.assertEqual(b''.join(c.buffer), b'{341}\r\nFrom: sender@host\r\nTo: recipient@domain\r\nSubject: booga booga boo\r\nContent-Type: multipart/alternative; boundary="----=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\r\n\r\n\r\n------=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\nSubject: this is subject text\r\nContent-Type: text/plain\r\n\r\n' + innerBody + b'\r\n------=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa--\r\n')
        return d.addCallback(cbProduced)

    def test_multiPartNoQuotes(self):
        """
        A boundary without does not have them added.
        """
        outerBody = b''
        innerBody = b'Contained body message text.  Squarge.'
        headers = OrderedDict()
        headers['from'] = 'sender@host'
        headers['to'] = 'recipient@domain'
        headers['subject'] = 'booga booga boo'
        headers['content-type'] = 'multipart/alternative; boundary=xyz'
        innerHeaders = OrderedDict()
        innerHeaders['subject'] = 'this is subject text'
        innerHeaders['content-type'] = 'text/plain'
        msg = FakeyMessage(headers, (), None, outerBody, 123, [FakeyMessage(innerHeaders, (), None, innerBody, None, None)])
        c = BufferingConsumer()
        p = imap4.MessageProducer(msg)
        d = p.beginProducing(c)

        def cbProduced(result):
            self.failUnlessIdentical(result, p)
            self.assertEqual(b''.join(c.buffer), b'{237}\r\nFrom: sender@host\r\nTo: recipient@domain\r\nSubject: booga booga boo\r\nContent-Type: multipart/alternative; boundary=xyz\r\n\r\n\r\n--xyz\r\nSubject: this is subject text\r\nContent-Type: text/plain\r\n\r\n' + innerBody + b'\r\n--xyz--\r\n')
        return d.addCallback(cbProduced)