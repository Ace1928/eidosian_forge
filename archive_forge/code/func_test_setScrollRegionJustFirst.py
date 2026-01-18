import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_setScrollRegionJustFirst(self):
    """
        With just a value for its C{first} argument,
        L{ServerProtocol.setScrollRegion} writes a control sequence with
        that parameter, a parameter separator, and finally a C{b'r'}.
        """
    self.protocol.setScrollRegion(first=1)
    self.assertEqual(self.transport.value(), self.CSI + b'1;' + b'r')