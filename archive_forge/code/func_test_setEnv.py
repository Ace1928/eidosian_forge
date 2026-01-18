import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
def test_setEnv(self):
    """
        When a client requests passing an environment variable, the
        SSHSession object should make the request by getting an
        ISessionSetEnv adapter for the avatar, then calling setEnv with the
        environment variable name and value.
        """
    self.assertFalse(self.session.requestReceived(b'env', common.NS(b'FAIL') + common.NS(b'bad')))
    self.assertIsInstance(self.session.session, StubSessionForStubAvatarWithEnv)
    self.assertRequestRaisedRuntimeError()
    self.assertFalse(self.session.requestReceived(b'env', common.NS(b'IGNORED') + common.NS(b'ignored')))
    self.assertEqual(self.flushLoggedErrors(), [])
    self.assertTrue(self.session.requestReceived(b'env', common.NS(b'NAME') + common.NS(b'value')))
    self.assertEqual(self.session.session.environ, {b'NAME': b'value'})