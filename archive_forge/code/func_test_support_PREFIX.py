import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_PREFIX(self):
    """
        The PREFIX support parameter is parsed into a dictionary mapping
        modes to two-tuples of status symbol and priority.
        """
    self._testFeatureDefault('PREFIX')
    self._testFeatureDefault('PREFIX', [('PREFIX', 'hello')])
    self.assertEqual(self._parseFeature('PREFIX', None), None)
    self.assertEqual(self._parseFeature('PREFIX', '(ohv)@%+'), {'o': ('@', 0), 'h': ('%', 1), 'v': ('+', 2)})
    self.assertEqual(self._parseFeature('PREFIX', '(hov)@%+'), {'o': ('%', 1), 'h': ('@', 0), 'v': ('+', 2)})