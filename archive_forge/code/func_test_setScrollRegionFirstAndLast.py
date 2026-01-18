import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_setScrollRegionFirstAndLast(self):
    """
        When given both C{first} and C{last}
        L{ServerProtocol.setScrollRegion} writes a control sequence with
        the first parameter, a parameter separator, the last
        parameter, and finally a C{b'r'}.
        """
    self.protocol.setScrollRegion(first=1, last=2)
    self.assertEqual(self.transport.value(), self.CSI + b'1;2' + b'r')