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
class FakeFile:
    """
    A fake file-like object that acts enough like a file for
    L{ConsoleUI.prompt}.
    """

    def __init__(self):
        self.inlines = []
        self.outchunks = []
        self.closed = False

    def readline(self):
        """
        Return a line from the 'inlines' list.
        """
        return self.inlines.pop(0)

    def write(self, chunk):
        """
        Append the given item to the 'outchunks' list.
        """
        if self.closed:
            raise OSError('the file was closed')
        self.outchunks.append(chunk)

    def close(self):
        """
        Set the 'closed' flag to True, explicitly marking that it has been
        closed.
        """
        self.closed = True