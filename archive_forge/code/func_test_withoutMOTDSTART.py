import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_withoutMOTDSTART(self):
    """
        If L{IRCClient} receives I{RPL_MOTD} and I{RPL_ENDOFMOTD} without
        receiving I{RPL_MOTDSTART}, L{IRCClient.receivedMOTD} is still
        called with a list of MOTD lines.
        """
    lines = [':host.name 372 nickname :- Welcome to host.name', ':host.name 376 nickname :End of /MOTD command.']
    for L in lines:
        self.client.dataReceived(L + '\r\n')
    self.assertEqual(self.client.calls, [('receivedMOTD', {'motd': ['Welcome to host.name']})])