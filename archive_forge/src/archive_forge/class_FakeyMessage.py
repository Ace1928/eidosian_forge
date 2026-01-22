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
@implementer(imap4.IMessage)
class FakeyMessage(util.FancyStrMixin):
    showAttributes = ('headers', 'flags', 'date', '_body', 'uid')

    def __init__(self, headers, flags, date, body, uid, subpart):
        self.headers = headers
        self.flags = flags
        self._body = body
        self.size = len(body)
        self.date = date
        self.uid = uid
        self.subpart = subpart

    def getHeaders(self, negate, *names):
        self.got_headers = (negate, names)
        return self.headers

    def getFlags(self):
        return self.flags

    def getInternalDate(self):
        return self.date

    def getBodyFile(self):
        return BytesIO(self._body)

    def getSize(self):
        return self.size

    def getUID(self):
        return self.uid

    def isMultipart(self):
        return self.subpart is not None

    def getSubPart(self, part):
        self.got_subpart = part
        return self.subpart[part]