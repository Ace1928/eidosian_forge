import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_setPrivateModes(self):
    """
        L{ServerProtocol.setPrivatesModes} writes a control sequence
        containing the requested private modes and ending in the
        L{CSFinalByte.SM}.
        """
    privateModesToSet = [privateModes.ERROR, privateModes.COLUMN, privateModes.ORIGIN]
    self.protocol.setModes(privateModesToSet)
    self.assertEqual(self.transport.value(), self.CSI + b';'.join((b'%d' % (m,) for m in privateModesToSet)) + CSFinalByte.SM.value)