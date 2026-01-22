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
def _messageSetSearchTest(self, queryTerms, expectedMessages):
    """
        Issue a search with given query and verify that the returned messages
        match the given expected messages.

        @param queryTerms: A string giving the search query.
        @param expectedMessages: A list of the message sequence numbers
            expected as the result of the search.
        @return: A L{Deferred} which fires when the test is complete.
        """

    def search():
        return self.client.search(queryTerms)
    d = self.connected.addCallback(strip(search))

    def searched(results):
        self.assertEqual(results, expectedMessages)
    d.addCallback(searched)
    d.addCallback(self._cbStopClient)
    d.addErrback(self._ebGeneral)
    self.loopback()
    return d