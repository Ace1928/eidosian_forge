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
def test_requestWindowChange(self):
    """
        When the client requests to change the window size, the SSHSession
        object should make the request by getting an ISession adapter for the
        avatar, then calling windowChanged with the new window size.
        """
    ret = self.session.requestReceived(b'window_change', session.packRequest_window_change((0, 0, 0, 0)))
    self.assertFalse(ret)
    self.assertRequestRaisedRuntimeError()
    self.assertSessionIsStubSession()
    self.assertTrue(self.session.requestReceived(b'window_change', session.packRequest_window_change((1, 2, 3, 4))))
    self.assertEqual(self.session.session.windowChange, (1, 2, 3, 4))