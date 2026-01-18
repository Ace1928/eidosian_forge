import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_parsePrefixParam(self):
    """
        L{ServerSupportedFeatures._parsePrefixParam} parses the ISUPPORT PREFIX
        parameter into a mapping from modes to prefix symbols, returns
        L{None} if there is no parseable prefix parameter or raises
        C{ValueError} if the prefix parameter is malformed.
        """
    _parsePrefixParam = irc.ServerSupportedFeatures._parsePrefixParam
    self.assertEqual(_parsePrefixParam(''), None)
    self.assertRaises(ValueError, _parsePrefixParam, 'hello')
    self.assertEqual(_parsePrefixParam('(ov)@+'), {'o': ('@', 0), 'v': ('+', 1)})