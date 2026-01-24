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
class NewFetchTests(TestCase, IMAP4HelperMixin):

    def setUp(self):
        self.received_messages = self.received_uid = None
        self.result = None
        self.server = imap4.IMAP4Server()
        self.server.state = 'select'
        self.server.mbox = self
        self.connected = defer.Deferred()
        self.client = SimpleClient(self.connected)

    def addListener(self, x):
        pass

    def removeListener(self, x):
        pass

    def fetch(self, messages, uid):
        self.received_messages = messages
        self.received_uid = uid
        return iter(zip(range(len(self.msgObjs)), self.msgObjs))

    def _fetchWork(self, uid):
        if uid:
            for i, msg in zip(range(len(self.msgObjs)), self.msgObjs):
                self.expected[i]['UID'] = str(msg.getUID())

        def result(R):
            self.result = R
        self.connected.addCallback(lambda _: self.function(self.messages, uid)).addCallback(result).addCallback(self._cbStopClient).addErrback(self._ebGeneral)
        d = loopback.loopbackTCP(self.server, self.client, noisy=False)
        d.addCallback(lambda x: self.assertEqual(self.result, self.expected))
        return d

    def testFetchUID(self):
        self.function = lambda m, u: self.client.fetchUID(m)
        self.messages = '7'
        self.msgObjs = [FakeyMessage({}, (), b'', b'', 12345, None), FakeyMessage({}, (), b'', b'', 999, None), FakeyMessage({}, (), b'', b'', 10101, None)]
        self.expected = {0: {'UID': '12345'}, 1: {'UID': '999'}, 2: {'UID': '10101'}}
        return self._fetchWork(0)

    def testFetchFlags(self, uid=0):
        self.function = self.client.fetchFlags
        self.messages = '9'
        self.msgObjs = [FakeyMessage({}, ['FlagA', 'FlagB', '\\FlagC'], b'', b'', 54321, None), FakeyMessage({}, ['\\FlagC', 'FlagA', 'FlagB'], b'', b'', 12345, None)]
        self.expected = {0: {'FLAGS': ['FlagA', 'FlagB', '\\FlagC']}, 1: {'FLAGS': ['\\FlagC', 'FlagA', 'FlagB']}}
        return self._fetchWork(uid)

    def testFetchFlagsUID(self):
        return self.testFetchFlags(1)

    def testFetchInternalDate(self, uid=0):
        self.function = self.client.fetchInternalDate
        self.messages = '13'
        self.msgObjs = [FakeyMessage({}, (), b'Fri, 02 Nov 2003 21:25:10 GMT', b'', 23232, None), FakeyMessage({}, (), b'Thu, 29 Dec 2013 11:31:52 EST', b'', 101, None), FakeyMessage({}, (), b'Mon, 10 Mar 1992 02:44:30 CST', b'', 202, None), FakeyMessage({}, (), b'Sat, 11 Jan 2000 14:40:24 PST', b'', 303, None)]
        self.expected = {0: {'INTERNALDATE': '02-Nov-2003 21:25:10 +0000'}, 1: {'INTERNALDATE': '29-Dec-2013 11:31:52 -0500'}, 2: {'INTERNALDATE': '10-Mar-1992 02:44:30 -0600'}, 3: {'INTERNALDATE': '11-Jan-2000 14:40:24 -0800'}}
        return self._fetchWork(uid)

    def testFetchInternalDateUID(self):
        return self.testFetchInternalDate(1)
    currentLocale = locale.setlocale(locale.LC_ALL, None)
    try:
        locale.setlocale(locale.LC_ALL, 'es_AR.UTF8')
    except locale.Error:
        noEsARLocale = True
    else:
        locale.setlocale(locale.LC_ALL, currentLocale)
        noEsARLocale = False

    @skipIf(noEsARLocale, 'The es_AR.UTF8 locale is not installed.')
    def test_fetchInternalDateLocaleIndependent(self):
        """
        The month name in the date is locale independent.
        """
        currentLocale = locale.setlocale(locale.LC_ALL, None)
        locale.setlocale(locale.LC_ALL, 'es_AR.UTF8')
        self.addCleanup(locale.setlocale, locale.LC_ALL, currentLocale)
        return self.testFetchInternalDate(1)

    def testFetchEnvelope(self, uid=0):
        self.function = self.client.fetchEnvelope
        self.messages = '15'
        self.msgObjs = [FakeyMessage({'from': 'user@domain', 'to': 'resu@domain', 'date': 'thursday', 'subject': 'it is a message', 'message-id': 'id-id-id-yayaya'}, (), b'', b'', 65656, None)]
        self.expected = {0: {'ENVELOPE': ['thursday', 'it is a message', [[None, None, 'user', 'domain']], [[None, None, 'user', 'domain']], [[None, None, 'user', 'domain']], [[None, None, 'resu', 'domain']], None, None, None, 'id-id-id-yayaya']}}
        return self._fetchWork(uid)

    def testFetchEnvelopeUID(self):
        return self.testFetchEnvelope(1)

    def test_fetchBodyStructure(self, uid=0):
        """
        L{IMAP4Client.fetchBodyStructure} issues a I{FETCH BODYSTRUCTURE}
        command and returns a Deferred which fires with a structure giving the
        result of parsing the server's response.  The structure is a list
        reflecting the parenthesized data sent by the server, as described by
        RFC 3501, section 7.4.2.
        """
        self.function = self.client.fetchBodyStructure
        self.messages = '3:9,10:*'
        self.msgObjs = [FakeyMessage({'content-type': 'text/plain; name=thing; key="value"', 'content-id': 'this-is-the-content-id', 'content-description': 'describing-the-content-goes-here!', 'content-transfer-encoding': '8BIT', 'content-md5': 'abcdef123456', 'content-disposition': 'attachment; filename=monkeys', 'content-language': 'es', 'content-location': 'http://example.com/monkeys'}, (), '', b'Body\nText\nGoes\nHere\n', 919293, None)]
        self.expected = {0: {'BODYSTRUCTURE': ['text', 'plain', ['key', 'value', 'name', 'thing'], 'this-is-the-content-id', 'describing-the-content-goes-here!', '8BIT', '20', '4', 'abcdef123456', ['attachment', ['filename', 'monkeys']], 'es', 'http://example.com/monkeys']}}
        return self._fetchWork(uid)

    def testFetchBodyStructureUID(self):
        """
        If passed C{True} for the C{uid} argument, C{fetchBodyStructure} can
        also issue a I{UID FETCH BODYSTRUCTURE} command.
        """
        return self.test_fetchBodyStructure(1)

    def test_fetchBodyStructureMultipart(self, uid=0):
        """
        L{IMAP4Client.fetchBodyStructure} can also parse the response to a
        I{FETCH BODYSTRUCTURE} command for a multipart message.
        """
        self.function = self.client.fetchBodyStructure
        self.messages = '3:9,10:*'
        innerMessage = FakeyMessage({'content-type': 'text/plain; name=thing; key="value"', 'content-id': 'this-is-the-content-id', 'content-description': 'describing-the-content-goes-here!', 'content-transfer-encoding': '8BIT', 'content-language': 'fr', 'content-md5': '123456abcdef', 'content-disposition': 'inline', 'content-location': 'outer space'}, (), b'', b'Body\nText\nGoes\nHere\n', 919293, None)
        self.msgObjs = [FakeyMessage({'content-type': 'multipart/mixed; boundary="xyz"', 'content-language': 'en', 'content-location': 'nearby'}, (), b'', b'', 919293, [innerMessage])]
        self.expected = {0: {'BODYSTRUCTURE': [['text', 'plain', ['key', 'value', 'name', 'thing'], 'this-is-the-content-id', 'describing-the-content-goes-here!', '8BIT', '20', '4', '123456abcdef', ['inline', None], 'fr', 'outer space'], 'mixed', ['boundary', 'xyz'], None, 'en', 'nearby']}}
        return self._fetchWork(uid)

    def testFetchSimplifiedBody(self, uid=0):
        self.function = self.client.fetchSimplifiedBody
        self.messages = '21'
        self.msgObjs = [FakeyMessage({}, (), b'', b'Yea whatever', 91825, [FakeyMessage({'content-type': 'image/jpg'}, (), b'', b'Body Body Body', None, None)])]
        self.expected = {0: {'BODY': [None, None, None, None, None, None, '12']}}
        return self._fetchWork(uid)

    def testFetchSimplifiedBodyUID(self):
        return self.testFetchSimplifiedBody(1)

    def testFetchSimplifiedBodyText(self, uid=0):
        self.function = self.client.fetchSimplifiedBody
        self.messages = '21'
        self.msgObjs = [FakeyMessage({'content-type': 'text/plain'}, (), b'', b'Yea whatever', 91825, None)]
        self.expected = {0: {'BODY': ['text', 'plain', None, None, None, None, '12', '1']}}
        return self._fetchWork(uid)

    def testFetchSimplifiedBodyTextUID(self):
        return self.testFetchSimplifiedBodyText(1)

    def testFetchSimplifiedBodyRFC822(self, uid=0):
        self.function = self.client.fetchSimplifiedBody
        self.messages = '21'
        self.msgObjs = [FakeyMessage({'content-type': 'message/rfc822'}, (), b'', b'Yea whatever', 91825, [FakeyMessage({'content-type': 'image/jpg'}, (), '', b'Body Body Body', None, None)])]
        self.expected = {0: {'BODY': ['message', 'rfc822', None, None, None, None, '12', [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], ['image', 'jpg', None, None, None, None, '14'], '1']}}
        return self._fetchWork(uid)

    def testFetchSimplifiedBodyRFC822UID(self):
        return self.testFetchSimplifiedBodyRFC822(1)

    def test_fetchSimplifiedBodyMultipart(self):
        """
        L{IMAP4Client.fetchSimplifiedBody} returns a dictionary mapping message
        sequence numbers to fetch responses for the corresponding messages.  In
        particular, for a multipart message, the value in the dictionary maps
        the string C{"BODY"} to a list giving the body structure information for
        that message, in the form of a list of subpart body structure
        information followed by the subtype of the message (eg C{"alternative"}
        for a I{multipart/alternative} message).  This structure is self-similar
        in the case where a subpart is itself multipart.
        """
        self.function = self.client.fetchSimplifiedBody
        self.messages = '21'
        singles = [FakeyMessage({'content-type': 'text/plain'}, (), b'date', b'Stuff', 54321, None), FakeyMessage({'content-type': 'text/html'}, (), b'date', b'Things', 32415, None)]
        alternative = FakeyMessage({'content-type': 'multipart/alternative'}, (), b'', b'Irrelevant', 12345, singles)
        mixed = FakeyMessage({'content-type': 'multipart/mixed'}, (), b'', b'RootOf', 98928, [alternative])
        self.msgObjs = [mixed]
        self.expected = {0: {'BODY': [[['text', 'plain', None, None, None, None, '5', '1'], ['text', 'html', None, None, None, None, '6', '1'], 'alternative'], 'mixed']}}
        return self._fetchWork(False)

    def testFetchMessage(self, uid=0):
        self.function = self.client.fetchMessage
        self.messages = '1,3,7,10101'
        self.msgObjs = [FakeyMessage({'Header': 'Value'}, (), b'', b'BODY TEXT\r\n', 91, None)]
        self.expected = {0: {'RFC822': 'Header: Value\r\n\r\nBODY TEXT\r\n'}}
        return self._fetchWork(uid)

    def testFetchMessageUID(self):
        return self.testFetchMessage(1)

    def testFetchHeaders(self, uid=0):
        self.function = self.client.fetchHeaders
        self.messages = '9,6,2'
        self.msgObjs = [FakeyMessage({'H1': 'V1', 'H2': 'V2'}, (), b'', b'', 99, None)]
        headers = nativeString(imap4._formatHeaders({'H1': 'V1', 'H2': 'V2'}))
        self.expected = {0: {'RFC822.HEADER': headers}}
        return self._fetchWork(uid)

    def testFetchHeadersUID(self):
        return self.testFetchHeaders(1)

    def testFetchBody(self, uid=0):
        self.function = self.client.fetchBody
        self.messages = '1,2,3,4,5,6,7'
        self.msgObjs = [FakeyMessage({'Header': 'Value'}, (), '', b'Body goes here\r\n', 171, None)]
        self.expected = {0: {'RFC822.TEXT': 'Body goes here\r\n'}}
        return self._fetchWork(uid)

    def testFetchBodyUID(self):
        return self.testFetchBody(1)

    def testFetchBodyParts(self):
        """
        Test the server's handling of requests for specific body sections.
        """
        self.function = self.client.fetchSpecific
        self.messages = '1'
        outerBody = ''
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
        self.msgObjs = [FakeyMessage(headers, (), None, outerBody, 123, [FakeyMessage(innerHeaders, (), None, innerBody1, None, None), FakeyMessage(innerHeaders2, (), None, innerBody2, None, None)])]
        self.expected = {0: [['BODY', ['1'], 'Contained body message text.  Squarge.']]}

        def result(R):
            self.result = R
        self.connected.addCallback(lambda _: self.function(self.messages, headerNumber=1))
        self.connected.addCallback(result)
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        d = loopback.loopbackTCP(self.server, self.client, noisy=False)
        d.addCallback(lambda ign: self.assertEqual(self.result, self.expected))
        return d

    def test_fetchBodyPartOfNonMultipart(self):
        """
        Single-part messages have an implicit first part which clients
        should be able to retrieve explicitly.  Test that a client
        requesting part 1 of a text/plain message receives the body of the
        text/plain part.
        """
        self.function = self.client.fetchSpecific
        self.messages = '1'
        parts = [1]
        outerBody = b'DA body'
        headers = OrderedDict()
        headers['from'] = 'sender@host'
        headers['to'] = 'recipient@domain'
        headers['subject'] = 'booga booga boo'
        headers['content-type'] = 'text/plain'
        self.msgObjs = [FakeyMessage(headers, (), None, outerBody, 123, None)]
        self.expected = {0: [['BODY', ['1'], 'DA body']]}

        def result(R):
            self.result = R
        self.connected.addCallback(lambda _: self.function(self.messages, headerNumber=parts))
        self.connected.addCallback(result)
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        d = loopback.loopbackTCP(self.server, self.client, noisy=False)
        d.addCallback(lambda ign: self.assertEqual(self.result, self.expected))
        return d

    def testFetchSize(self, uid=0):
        self.function = self.client.fetchSize
        self.messages = '1:100,2:*'
        self.msgObjs = [FakeyMessage({}, (), b'', b'x' * 20, 123, None)]
        self.expected = {0: {'RFC822.SIZE': '20'}}
        return self._fetchWork(uid)

    def testFetchSizeUID(self):
        return self.testFetchSize(1)

    def testFetchFull(self, uid=0):
        self.function = self.client.fetchFull
        self.messages = '1,3'
        self.msgObjs = [FakeyMessage({}, ('\\XYZ', '\\YZX', 'Abc'), b'Sun, 25 Jul 2010 06:20:30 -0400 (EDT)', b'xyz' * 2, 654, None), FakeyMessage({}, ('\\One', '\\Two', 'Three'), b'Mon, 14 Apr 2003 19:43:44 -0400', b'abc' * 4, 555, None)]
        self.expected = {0: {'FLAGS': ['\\XYZ', '\\YZX', 'Abc'], 'INTERNALDATE': '25-Jul-2010 06:20:30 -0400', 'RFC822.SIZE': '6', 'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'BODY': [None, None, None, None, None, None, '6']}, 1: {'FLAGS': ['\\One', '\\Two', 'Three'], 'INTERNALDATE': '14-Apr-2003 19:43:44 -0400', 'RFC822.SIZE': '12', 'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'BODY': [None, None, None, None, None, None, '12']}}
        return self._fetchWork(uid)

    def testFetchFullUID(self):
        return self.testFetchFull(1)

    def testFetchAll(self, uid=0):
        self.function = self.client.fetchAll
        self.messages = '1,2:3'
        self.msgObjs = [FakeyMessage({}, (), b'Mon, 14 Apr 2003 19:43:44 +0400', b'Lalala', 10101, None), FakeyMessage({}, (), b'Tue, 15 Apr 2003 19:43:44 +0200', b'Alalal', 20202, None)]
        self.expected = {0: {'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'RFC822.SIZE': '6', 'INTERNALDATE': '14-Apr-2003 19:43:44 +0400', 'FLAGS': []}, 1: {'ENVELOPE': [None, None, [[None, None, None]], [[None, None, None]], None, None, None, None, None, None], 'RFC822.SIZE': '6', 'INTERNALDATE': '15-Apr-2003 19:43:44 +0200', 'FLAGS': []}}
        return self._fetchWork(uid)

    def testFetchAllUID(self):
        return self.testFetchAll(1)

    def testFetchFast(self, uid=0):
        self.function = self.client.fetchFast
        self.messages = '1'
        self.msgObjs = [FakeyMessage({}, ('\\X',), b'19 Mar 2003 19:22:21 -0500', b'', 9, None)]
        self.expected = {0: {'FLAGS': ['\\X'], 'INTERNALDATE': '19-Mar-2003 19:22:21 -0500', 'RFC822.SIZE': '0'}}
        return self._fetchWork(uid)

    def testFetchFastUID(self):
        return self.testFetchFast(1)