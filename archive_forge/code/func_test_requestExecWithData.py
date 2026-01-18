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
def test_requestExecWithData(self):
    """
        When a client executes a command, it should be able to give pass data
        back and forth.
        """
    ret = self.session.requestReceived(b'exec', common.NS(b'repeat hello'))
    self.assertTrue(ret)
    self.assertSessionIsStubSession()
    self.session.dataReceived(b'some data')
    self.assertEqual(self.session.session.execTransport.data, b'some data')
    self.assertEqual(self.session.conn.data[self.session], [b'hello', b'some data', b'\r\n'])
    self.session.eofReceived()
    self.session.closeReceived()
    self.session.closed()
    self.assertTrue(self.session.session.execTransport.closed)
    self.assertEqual(self.session.conn.requests[self.session], [(b'exit-status', b'\x00\x00\x00\x00', False)])