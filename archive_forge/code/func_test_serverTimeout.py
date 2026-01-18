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
def test_serverTimeout(self):
    """
        The *client* has a timeout mechanism which will close connections that
        are inactive for a period.
        """
    c = Clock()
    self.server.timeoutTest = True
    self.client.timeout = 5
    self.client.callLater = c.callLater
    self.selectedArgs = None

    def login():
        d = self.client.login(b'testuser', b'password-test')
        c.advance(5)
        d.addErrback(timedOut)
        return d

    def timedOut(failure):
        self._cbStopClient(None)
        failure.trap(error.TimeoutError)
    d = self.connected.addCallback(strip(login))
    d.addErrback(self._ebGeneral)
    return defer.gatherResults([d, self.loopback()])