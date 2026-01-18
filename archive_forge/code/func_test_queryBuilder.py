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
def test_queryBuilder(self):
    inputs = [imap4.Query(flagged=1), imap4.Query(sorted=1, unflagged=1, deleted=1), imap4.Or(imap4.Query(flagged=1), imap4.Query(deleted=1)), imap4.Query(before='today'), imap4.Or(imap4.Query(deleted=1), imap4.Query(unseen=1), imap4.Query(new=1)), imap4.Or(imap4.Not(imap4.Or(imap4.Query(sorted=1, since='yesterday', smaller=1000), imap4.Query(sorted=1, before='tuesday', larger=10000), imap4.Query(sorted=1, unseen=1, deleted=1, before='today'), imap4.Not(imap4.Query(subject='spam')))), imap4.Not(imap4.Query(uid='1:5')))]
    outputs = ['FLAGGED', '(DELETED UNFLAGGED)', '(OR FLAGGED DELETED)', '(BEFORE "today")', '(OR DELETED (OR UNSEEN NEW))', '(OR (NOT (OR (SINCE "yesterday" SMALLER 1000) (OR (BEFORE "tuesday" LARGER 10000) (OR (BEFORE "today" DELETED UNSEEN) (NOT (SUBJECT "spam")))))) (NOT (UID 1:5)))']
    for query, expected in zip(inputs, outputs):
        self.assertEqual(query, expected)