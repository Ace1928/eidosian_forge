import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommandWithTags(self):
    """
        Passing a command and parameters with a specified prefix and tags
        to L{IRC.sendCommand} results in a proper query string including the
        specified line prefix and appropriate tags syntax.  The query string
        should be output as follows:
        @tags :prefix COMMAND param1 param2\r

        The tags are a string of IRCv3 tags, preceded by '@'.  The rest
        of the string is as described in test_sendMessage.  For more on
        the message tag format, see U{the IRCv3 specification
        <https://ircv3.net/specs/core/message-tags-3.2.html>}.
        """
    sendTags = {'aaa': 'bbb', 'ccc': None, 'example.com/ddd': 'eee'}
    expectedTags = (b'aaa=bbb', b'ccc', b'example.com/ddd=eee')
    self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
    outMsg = self.f.getvalue()
    outTagStr, outLine = outMsg.split(b' ', 1)
    outTags = outTagStr[1:].split(b';')
    self.assertEqual(outLine, b':irc.example.com CMD param1 param2\r\n')
    self.assertEqual(sorted(expectedTags), sorted(outTags))