import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _parseFeature(self, name, value=None):
    """
        Parse a feature, with the given name and value, according to the
        ISUPPORT specifications and return the parsed value.
        """
    supported = self._parse([(name, value)])
    return supported.getFeature(name)