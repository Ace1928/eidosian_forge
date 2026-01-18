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
def test_setEnvNotProvidingISessionSetEnv(self):
    """
        If the avatar does not have an ISessionSetEnv adapter, then a
        request to pass an environment variable fails gracefully.
        """
    self.doCleanups()
    self.setUp(register_adapters=False)
    components.registerAdapter(StubSessionForStubAvatar, StubAvatar, session.ISession)
    self.assertFalse(self.session.requestReceived(b'env', common.NS(b'NAME') + common.NS(b'value')))