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
def test_setEnvSessionShare(self):
    """
        Multiple setenv requests will share the same session.
        """
    test_session = self.getSSHSession()
    self.assertTrue(test_session.requestReceived(b'env', common.NS(b'Key1') + common.NS(b'Value 1')))
    self.assertTrue(test_session.requestReceived(b'env', common.NS(b'Key2') + common.NS(b'Value2')))
    self.assertIsInstance(test_session.session, StubSessionForStubAvatarWithEnv)
    self.assertEqual({b'Key1': b'Value 1', b'Key2': b'Value2'}, test_session.session.environ)