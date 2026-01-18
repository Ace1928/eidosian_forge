import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_reverseIndex(self):
    """
        L{ServerProtocol.reverseIndex} writes the control sequence
        ending in the L{C1SevenBit.RI}.
        """
    self.protocol.reverseIndex()
    self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.RI.value)