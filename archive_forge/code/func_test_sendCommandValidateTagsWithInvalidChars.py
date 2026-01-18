import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommandValidateTagsWithInvalidChars(self):
    """
        Passing a tag name containing invalid characters to L{IRC.sendCommand}
        raises a C{ValueError}.
        """
    sendTags = {'aaa_b^@': 'ccc'}
    error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
    self.assertEqual(error.args[0], 'Tag contains invalid characters.')