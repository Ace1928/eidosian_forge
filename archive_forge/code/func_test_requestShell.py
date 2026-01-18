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
def test_requestShell(self):
    """
        When a client requests a shell, the SSHSession object should get
        the shell by getting an ISession adapter for the avatar, then
        calling openShell() with a ProcessProtocol to attach.
        """
    ret = self.session.requestReceived(b'shell', b'')
    self.assertTrue(ret)
    self.assertSessionIsStubSession()
    self.assertIsInstance(self.session.client, session.SSHSessionProcessProtocol)
    self.assertIs(self.session.session.shellProtocol, self.session.client)
    self.assertFalse(self.session.requestReceived(b'shell', b''))
    self.assertRequestRaisedRuntimeError()