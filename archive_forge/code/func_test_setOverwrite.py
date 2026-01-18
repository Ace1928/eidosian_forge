import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_setOverwrite(self):
    """
        When local file already exists it can be overwritten using the
        L{DccFileReceive.set_overwrite} method.
        """
    fp = FilePath(self.mktemp())
    fp.setContent(b'I love contributing to Twisted!')
    protocol = self.makeConnectedDccFileReceive(fp.path, overwrite=True)
    self.allDataReceivedForProtocol(protocol, b'Twisted rocks!')
    self.assertEqual(fp.getContent(), b'Twisted rocks!')