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
def test_packRequest_pty_req(self):
    """
        See test_parseRequest_pty_req for the payload format.
        """
    packed = session.packRequest_pty_req(b'xterm', (2, 1, 3, 4), b'\x05\x00\x00\x00\x06')
    self.assertEqual(packed, common.NS(b'xterm') + struct.pack('>4L', 1, 2, 3, 4) + common.NS(struct.pack('>BL', 5, 6)))