import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_MAXLIST(self):
    """
        The MAXLIST support parameter is parsed into a sequence of two-tuples
        giving modes and their limits.
        """
    self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50'), [('b', 25), ('eI', 50)])
    self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50,a:3.1415'), [('b', 25), ('eI', 50), ('a', None)])
    self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50,a:notanint'), [('b', 25), ('eI', 50), ('a', None)])