import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_SAFELIST(self):
    """
        The SAFELIST support parameter is parsed into a boolean indicating
        whether the safe "list" command is supported or not.
        """
    self.assertEqual(self._parseFeature('SAFELIST'), True)