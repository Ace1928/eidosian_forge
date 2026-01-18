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
def test_parseIdList(self):
    """
        The function to parse sequence ranges yields appropriate L{MessageSet}
        objects.
        """
    inputs = [b'1:*', b'5:*', b'1:2,5:*', b'*', b'1', b'1,2', b'1,3,5', b'1:10', b'1:10,11', b'1:5,10:20', b'1,5:10', b'1,5:10,15:20', b'1:10,15,20:25', b'4:2']
    outputs = [MessageSet(1, None), MessageSet(5, None), MessageSet(5, None) + MessageSet(1, 2), MessageSet(None, None), MessageSet(1), MessageSet(1, 2), MessageSet(1) + MessageSet(3) + MessageSet(5), MessageSet(1, 10), MessageSet(1, 11), MessageSet(1, 5) + MessageSet(10, 20), MessageSet(1) + MessageSet(5, 10), MessageSet(1) + MessageSet(5, 10) + MessageSet(15, 20), MessageSet(1, 10) + MessageSet(15) + MessageSet(20, 25), MessageSet(2, 4)]
    lengths = [None, None, None, 1, 1, 2, 3, 10, 11, 16, 7, 13, 17, 3]
    for input, expected in zip(inputs, outputs):
        self.assertEqual(imap4.parseIdList(input), expected)
    for input, expected in zip(inputs, lengths):
        if expected is None:
            self.assertRaises(TypeError, len, imap4.parseIdList(input))
        else:
            L = len(imap4.parseIdList(input))
            self.assertEqual(L, expected, f'len({input!r}) = {L!r} != {expected!r}')