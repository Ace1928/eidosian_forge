import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_sendCommandValidateTagValueEscaping(self):
    """
        Tags with values containing invalid characters passed to
        L{IRC.sendCommand} are escaped.
        """
    sendTags = {'aaa': 'bbb', 'ccc': 'test\r\n \\;;'}
    expectedTags = (b'aaa=bbb', b'ccc=test\\r\\n\\s\\\\\\:\\:')
    self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
    outMsg = self.f.getvalue()
    outTagStr, outLine = outMsg.split(b' ', 1)
    outTags = outTagStr[1:].split(b';')
    self.assertEqual(sorted(outTags), sorted(expectedTags))