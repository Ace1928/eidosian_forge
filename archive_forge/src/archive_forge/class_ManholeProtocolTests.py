import sys
import traceback
from typing import Optional
from twisted.conch import manhole
from twisted.conch.insults import insults
from twisted.conch.test.test_recvline import (
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.trial import unittest
class ManholeProtocolTests(unittest.TestCase):
    """
    Tests for L{manhole.Manhole}.
    """

    def test_interruptResetsInterpreterBuffer(self):
        """
        L{manhole.Manhole.handle_INT} should cause the interpreter input buffer
        to be reset.
        """
        transport = StringTransport()
        terminal = insults.ServerProtocol(manhole.Manhole)
        terminal.makeConnection(transport)
        protocol = terminal.terminalProtocol
        interpreter = protocol.interpreter
        interpreter.buffer.extend(['1', '2'])
        protocol.handle_INT()
        self.assertFalse(interpreter.buffer)