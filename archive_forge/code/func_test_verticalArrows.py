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
def test_verticalArrows(self):
    """
        When L{HistoricRecvLine} receives UP_ARROW or DOWN_ARROW
        keystrokes it move the current index in the current history
        buffer up or down, and resets the current line buffer to the
        previous or next line in history, respectively for each.
        """
    kR = lambda ch: self.p.keystrokeReceived(ch, None)
    for ch in iterbytes(b'xyz\nabc\n123\n'):
        kR(ch)
    self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))
    self.assertEqual(self.p.currentLineBuffer(), (b'', b''))
    kR(self.pt.UP_ARROW)
    self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc'), (b'123',)))
    self.assertEqual(self.p.currentLineBuffer(), (b'123', b''))
    kR(self.pt.UP_ARROW)
    self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz',), (b'abc', b'123')))
    self.assertEqual(self.p.currentLineBuffer(), (b'abc', b''))
    kR(self.pt.UP_ARROW)
    self.assertEqual(self.p.currentHistoryBuffer(), ((), (b'xyz', b'abc', b'123')))
    self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
    kR(self.pt.UP_ARROW)
    self.assertEqual(self.p.currentHistoryBuffer(), ((), (b'xyz', b'abc', b'123')))
    self.assertEqual(self.p.currentLineBuffer(), (b'xyz', b''))
    for i in range(4):
        kR(self.pt.DOWN_ARROW)
    self.assertEqual(self.p.currentHistoryBuffer(), ((b'xyz', b'abc', b'123'), ()))