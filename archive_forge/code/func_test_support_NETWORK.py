import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_NETWORK(self):
    """
        The NETWORK support parameter is parsed as the network name, as
        specified by the server.
        """
    self.assertEqual(self._parseFeature('NETWORK', 'IRCNet'), 'IRCNet')