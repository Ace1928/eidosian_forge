import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_support_CHANNELLEN(self):
    """
        The CHANNELLEN support parameter is parsed into an integer value
        indicating the maximum channel name length, otherwise, if the
        parameter is missing or invalid, the default value (as specified by
        RFC 1459) is used.
        """
    default = irc.ServerSupportedFeatures()._features['CHANNELLEN']
    self._testIntOrDefaultFeature('CHANNELLEN', default)