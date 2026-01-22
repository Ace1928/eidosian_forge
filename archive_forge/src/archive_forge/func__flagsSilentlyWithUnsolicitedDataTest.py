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
def _flagsSilentlyWithUnsolicitedDataTest(self, method, item):
    """
        Test unsolicited data received in response to a silent flag modifying
        method.  Call the method, assert that the correct bytes are sent,
        deliver the unsolicited I{FETCH} response, and assert that the result
        of the Deferred returned by the method is correct.

        @param method: The name of the method to test.
        @param item: The data item which is expected to be specified.
        """
    d = getattr(self.client, method)('3', ('\\Read', '\\Seen'), True)
    self.assertEqual(self.transport.value(), b'0001 STORE 3 ' + item + b' (\\Read \\Seen)\r\n')
    self.client.lineReceived(b'* 2 FETCH (FLAGS (\\Read \\Seen))')
    self.client.lineReceived(b'0001 OK STORE completed')
    self.assertEqual(self.successResultOf(d), {})
    self.assertEqual(self.client.flags, {2: ['\\Read', '\\Seen']})