import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_shiftInAfterApplicationData(self):
    """
        Application data bytes followed by a shift-in command are passed to a
        call to C{write} before the terminal's C{shiftIn} method is called.
        """
    self._applicationDataTest(b'ab\x15', [('write', (b'ab',)), ('shiftIn',)])