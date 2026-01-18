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
def test_seq_rangeExamples(self):
    """
        Test the C{seq-range} examples from Section 9, "Formal Syntax"
        of RFC 3501::

            Example: 2:4 and 4:2 are equivalent and indicate values
                     2, 3, and 4.

            Example: a unique identifier sequence range of
                     3291:* includes the UID of the last message in
                     the mailbox, even if that value is less than 3291.

        @see: U{http://tools.ietf.org/html/rfc3501#section-9}
        """
    self.assertEqual(MessageSet(2, 4), MessageSet(4, 2))
    self.assertEqual(list(MessageSet(2, 4)), [2, 3, 4])
    m = MessageSet(3291, None)
    m.last = 3290
    self.assertEqual(list(m), [3290, 3291])