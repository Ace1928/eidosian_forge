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
def test_promptRepeatedly(self):
    """
        L{ConsoleUI.prompt} writes a message to the console, then reads a line.
        If that line is neither 'yes' nor 'no', then it says "Please enter
        'yes' or 'no'" until it gets a 'yes' or a 'no', at which point it
        returns a Deferred that answers either True or False.
        """
    self.newFile([b'what', b'uh', b'okay', b'yes'])
    l = []
    self.ui.prompt(b'Please say something useful.').addCallback(l.append)
    self.assertEqual([True], l)
    self.assertEqual(self.fakeFile.outchunks, [b'Please say something useful.'] + [b"Please type 'yes' or 'no': "] * 3)
    self.assertTrue(self.fakeFile.closed)
    self.newFile([b'blah', b'stuff', b'feh', b'no'])
    l = []
    self.ui.prompt(b'Please say something negative.').addCallback(l.append)
    self.assertEqual([False], l)
    self.assertEqual(self.fakeFile.outchunks, [b'Please say something negative.'] + [b"Please type 'yes' or 'no': "] * 3)
    self.assertTrue(self.fakeFile.closed)