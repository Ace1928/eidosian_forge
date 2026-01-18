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
def test_parseRequest_pty_req(self):
    """
        The payload of a pty-req message is::
            string  terminal
            uint32  columns
            uint32  rows
            uint32  x pixels
            uint32  y pixels
            string  modes

        Modes are::
            byte    mode number
            uint32  mode value
        """
    self.assertEqual(session.parseRequest_pty_req(common.NS(b'xterm') + struct.pack('>4L', 1, 2, 3, 4) + common.NS(struct.pack('>BL', 5, 6))), (b'xterm', (2, 1, 3, 4), [(5, 6)]))