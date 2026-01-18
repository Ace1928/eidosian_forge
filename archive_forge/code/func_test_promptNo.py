import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
def test_promptNo(self):
    """
        L{ConsoleUI.prompt} writes a message to the console, then reads a line.
        If that line is 'no', then it returns a L{Deferred} that fires with
        False.
        """
    for okNo in [b'no', b'No', b'no\n']:
        self.newFile([okNo])
        l = []
        self.ui.prompt('Goodbye, world!').addCallback(l.append)
        self.assertEqual(['Goodbye, world!'], self.fakeFile.outchunks)
        self.assertEqual([False], l)
        self.assertTrue(self.fakeFile.closed)