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
@implementer(imap4.ISearchableMailbox)
class FetchSearchStoreTests(TestCase, IMAP4HelperMixin):

    def setUp(self):
        self.expected = self.result = None
        self.server_received_query = None
        self.server_received_uid = None
        self.server_received_parts = None
        self.server_received_messages = None
        self.server = imap4.IMAP4Server()
        self.server.state = 'select'
        self.server.mbox = self
        self.connected = defer.Deferred()
        self.client = SimpleClient(self.connected)

    def search(self, query, uid):
        if query == [b'FOO']:
            raise imap4.IllegalQueryError('FOO is not a valid search criteria')
        self.server_received_query = query
        self.server_received_uid = uid
        return self.expected

    def addListener(self, *a, **kw):
        pass
    removeListener = addListener

    def _searchWork(self, uid):

        def search():
            return self.client.search(self.query, uid=uid)

        def result(R):
            self.result = R
        self.connected.addCallback(strip(search)).addCallback(result).addCallback(self._cbStopClient).addErrback(self._ebGeneral)

        def check(ignored):
            self.assertFalse(self.result is self.expected)
            self.assertEqual(self.result, self.expected)
            self.assertEqual(self.uid, self.server_received_uid)
            self.assertEqual(imap4.parseNestedParens(self.query.encode('charmap')), self.server_received_query)
        d = loopback.loopbackTCP(self.server, self.client, noisy=False)
        d.addCallback(check)
        return d

    def testSearch(self):
        self.query = imap4.Or(imap4.Query(header=('subject', 'substring')), imap4.Query(larger=1024, smaller=4096))
        self.expected = [1, 4, 5, 7]
        self.uid = 0
        return self._searchWork(0)

    def testUIDSearch(self):
        self.query = imap4.Or(imap4.Query(header=('subject', 'substring')), imap4.Query(larger=1024, smaller=4096))
        self.uid = 1
        self.expected = [1, 2, 3]
        return self._searchWork(1)

    def getUID(self, msg):
        try:
            return self.expected[msg]['UID']
        except (TypeError, IndexError):
            return self.expected[msg - 1]
        except KeyError:
            return 42

    def fetch(self, messages, uid):
        self.server_received_uid = uid
        self.server_received_messages = str(messages)
        return self.expected

    def _fetchWork(self, fetch):

        def result(R):
            self.result = R
        self.connected.addCallback(strip(fetch)).addCallback(result).addCallback(self._cbStopClient).addErrback(self._ebGeneral)

        def check(ignored):
            self.assertFalse(self.result is self.expected)
            self.parts and self.parts.sort()
            self.server_received_parts and self.server_received_parts.sort()
            if self.uid:
                for k, v in self.expected.items():
                    v['UID'] = str(k)
            self.assertEqual(self.result, self.expected)
            self.assertEqual(self.uid, self.server_received_uid)
            self.assertEqual(self.parts, self.server_received_parts)
            self.assertEqual(imap4.parseIdList(self.messages), imap4.parseIdList(self.server_received_messages))
        d = loopback.loopbackTCP(self.server, self.client, noisy=False)
        d.addCallback(check)
        return d

    def test_invalidTerm(self):
        """
        If, as part of a search, an ISearchableMailbox raises an
        IllegalQueryError (e.g. due to invalid search criteria), client sees a
        failure response, and an IllegalQueryError is logged on the server.
        """
        query = 'FOO'

        def search():
            return self.client.search(query)
        d = self.connected.addCallback(strip(search))
        d = self.assertFailure(d, imap4.IMAP4Exception)

        def errorReceived(results):
            """
            Verify that the server logs an IllegalQueryError and the
            client raises an IMAP4Exception with 'Search failed:...'
            """
            self.client.transport.loseConnection()
            self.server.transport.loseConnection()
            errors = self.flushLoggedErrors(imap4.IllegalQueryError)
            self.assertEqual(len(errors), 1)
            self.assertEqual(str(b'SEARCH failed: FOO is not a valid search criteria'), str(results))
        d.addCallback(errorReceived)
        d.addErrback(self._ebGeneral)
        self.loopback()
        return d