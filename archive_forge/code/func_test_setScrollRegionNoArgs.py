import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_setScrollRegionNoArgs(self):
    """
        With no arguments, L{ServerProtocol.setScrollRegion} writes a
        control sequence with no parameters, but a parameter
        separator, and ending in C{b'r'}.
        """
    self.protocol.setScrollRegion()
    self.assertEqual(self.transport.value(), self.CSI + b';' + b'r')