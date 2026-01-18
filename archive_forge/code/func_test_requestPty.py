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
def test_requestPty(self):
    """
        When a client requests a PTY, the SSHSession object should make
        the request by getting an ISession adapter for the avatar, then
        calling getPty with the terminal type, the window size, and any modes
        the client gave us.
        """
    self.doCleanups()
    self.setUp(register_adapters=False)
    components.registerAdapter(StubSessionForStubAvatar, StubAvatar, session.ISession)
    test_session = self.getSSHSession()
    ret = test_session.requestReceived(b'pty_req', session.packRequest_pty_req(b'bad', (1, 2, 3, 4), b''))
    self.assertFalse(ret)
    self.assertIsInstance(test_session.session, StubSessionForStubAvatar)
    self.assertRequestRaisedRuntimeError()
    self.assertTrue(test_session.requestReceived(b'pty_req', session.packRequest_pty_req(b'good', (1, 2, 3, 4), b'')))
    self.assertEqual(test_session.session.ptyRequest, (b'good', (1, 2, 3, 4), []))