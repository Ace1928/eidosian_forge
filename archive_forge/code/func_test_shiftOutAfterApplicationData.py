import textwrap
from typing import Optional, Type
from twisted.conch.insults.insults import (
from twisted.internet.protocol import Protocol
from twisted.internet.testing import StringTransport
from twisted.python.compat import iterbytes
from twisted.python.constants import ValueConstant, Values
from twisted.trial import unittest
def test_shiftOutAfterApplicationData(self):
    """
        Application data bytes followed by a shift-out command are passed to a
        call to C{write} before the terminal's C{shiftOut} method is called.
        """
    self._applicationDataTest(b'ab\x14', [('write', (b'ab',)), ('shiftOut',)])