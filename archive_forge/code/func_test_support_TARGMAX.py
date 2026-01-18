import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_TARGMAX(self):
    """
        The TARGMAX support parameter is parsed into a dictionary, mapping
        strings to integers, of the maximum number of targets for a particular
        command.
        """
    self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3'), {'PRIVMSG': 4, 'NOTICE': 3})
    self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3,KICK:3.1415'), {'PRIVMSG': 4, 'NOTICE': 3, 'KICK': None})
    self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3,KICK:notanint'), {'PRIVMSG': 4, 'NOTICE': 3, 'KICK': None})