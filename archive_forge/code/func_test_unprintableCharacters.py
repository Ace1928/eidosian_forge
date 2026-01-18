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
def test_unprintableCharacters(self):
    """
        When L{HistoricRecvLine} receives a keystroke for an unprintable
        function key with no assigned behavior, the line buffer is unmodified.
        """
    kR = lambda ch: self.p.keystrokeReceived(ch, None)
    pt = self.pt
    for ch in (pt.F1, pt.F2, pt.F3, pt.F4, pt.F5, pt.F6, pt.F7, pt.F8, pt.F9, pt.F10, pt.F11, pt.F12, pt.PGUP, pt.PGDN):
        kR(ch)
        self.assertEqual(self.p.currentLineBuffer(), (b'', b''))