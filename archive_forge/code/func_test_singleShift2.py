import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_singleShift2(self):
    """
        L{ServerProtocol.singleShift2} writes an escape sequence
        followed by L{C1SevenBit.SS2}
        """
    self.protocol.singleShift2()
    self.assertEqual(self.transport.value(), self.ESC + C1SevenBit.SS2.value)