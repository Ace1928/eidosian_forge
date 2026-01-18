import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_selectGraphicRendition(self):
    """
        L{ServerProtocol.selectGraphicRendition} writes a control
        sequence containing the requested attributes and ending with
        L{CSFinalByte.SGR}
        """
    self.protocol.selectGraphicRendition(str(BLINK), str(UNDERLINE))
    self.assertEqual(self.transport.value(), self.CSI + b'%d;%d' % (BLINK, UNDERLINE) + CSFinalByte.SGR.value)