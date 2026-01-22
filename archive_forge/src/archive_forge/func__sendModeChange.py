import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _sendModeChange(self, msg, args='', target=None):
    """
        Build a MODE string and send it to the client.
        """
    if target is None:
        target = '#chan'
    message = f':Wolf!~wolf@yok.utu.fi MODE {target} {msg} {args}\r\n'
    self.client.dataReceived(message)