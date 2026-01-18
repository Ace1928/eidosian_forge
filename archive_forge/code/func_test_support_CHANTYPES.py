import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_CHANTYPES(self):
    """
        The CHANTYPES support parameter is parsed into a tuple of
        valid channel prefix characters.
        """
    self._testFeatureDefault('CHANTYPES')
    self.assertEqual(self._parseFeature('CHANTYPES', '#&%'), ('#', '&', '%'))