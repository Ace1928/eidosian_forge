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
def assertCommandExceptionResponse(self, exception, tag, expectedResponse):
    """
        Assert that the given exception results in the expected
        response.

        @param exception: The exception to raise.
        @type exception: L{Exception}

        @param: The IMAP tag.

        @type: L{bytes}

        @param expectedResponse: The expected bad response.
        @type expectedResponse: L{bytes}
        """

    def raises(serverInstance, tag, user, passwd):
        raise exception
    self.assertEqual(self.server.state, 'unauth')
    self.server.unauth_LOGIN = (raises,) + self.server.unauth_LOGIN[1:]
    self.server.dispatchCommand(tag, b'LOGIN', b'user passwd')
    self.assertEqual(self.transport.value(), b' '.join([tag, expectedResponse]))