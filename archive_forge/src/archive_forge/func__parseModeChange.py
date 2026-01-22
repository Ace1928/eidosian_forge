import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _parseModeChange(self, results, target=None):
    """
        Parse the results, do some test and return the data to check.
        """
    if target is None:
        target = '#chan'
    for n, result in enumerate(results):
        method, data = result
        self.assertEqual(method, 'modeChanged')
        self.assertEqual(data['user'], 'Wolf!~wolf@yok.utu.fi')
        self.assertEqual(data['channel'], target)
        results[n] = tuple((data[key] for key in ('set', 'modes', 'args')))
    return results