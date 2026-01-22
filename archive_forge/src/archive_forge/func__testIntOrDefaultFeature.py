import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _testIntOrDefaultFeature(self, name, default=None):
    """
        Perform some common tests on a feature known to use L{_intOrDefault}.
        """
    self.assertEqual(self._parseFeature(name, None), default)
    self.assertEqual(self._parseFeature(name, 'notanint'), default)
    self.assertEqual(self._parseFeature(name, '42'), 42)