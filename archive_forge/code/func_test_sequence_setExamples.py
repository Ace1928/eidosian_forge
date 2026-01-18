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
def test_sequence_setExamples(self):
    """
        Test the C{sequence-set} examples from Section 9, "Formal
        Syntax" of RFC 3501.  In particular, L{MessageSet} reorders
        and coalesces overlaps::

            Example: a message sequence number set of
                     2,4:7,9,12:* for a mailbox with 15 messages is
                     equivalent to 2,4,5,6,7,9,12,13,14,15

            Example: a message sequence number set of *:4,5:7
                     for a mailbox with 10 messages is equivalent to
                     10,9,8,7,6,5,4,5,6,7 and MAY be reordered and
                     overlap coalesced to be 4,5,6,7,8,9,10.

        @see: U{http://tools.ietf.org/html/rfc3501#section-9}
        """
    fromFifteenMessages = MessageSet(2) + MessageSet(4, 7) + MessageSet(9) + MessageSet(12, None)
    fromFifteenMessages.last = 15
    self.assertEqual(','.join((str(i) for i in fromFifteenMessages)), '2,4,5,6,7,9,12,13,14,15')
    fromTenMessages = MessageSet(None, 4) + MessageSet(5, 7)
    fromTenMessages.last = 10
    self.assertEqual(','.join((str(i) for i in fromTenMessages)), '4,5,6,7,8,9,10')