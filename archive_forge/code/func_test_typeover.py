import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
def test_typeover(self):
    """
        When in INSERT mode and upon receiving a keystroke with a printable
        character, L{HistoricRecvLine} replaces the character at
        the cursor with the typed character rather than inserting before.
        Ah, the ironies of INSERT mode.
        """
    kR = lambda ch: self.p.keystrokeReceived(ch, None)
    for ch in iterbytes(b'xyz'):
        kR(ch)
    kR(self.pt.INSERT)
    kR(self.pt.LEFT_ARROW)
    kR(b'A')
    self.assertEqual(self.p.currentLineBuffer(), (b'xyA', b''))
    kR(self.pt.LEFT_ARROW)
    kR(b'B')
    self.assertEqual(self.p.currentLineBuffer(), (b'xyB', b''))