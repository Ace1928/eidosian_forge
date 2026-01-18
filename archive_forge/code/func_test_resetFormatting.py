import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_resetFormatting(self):
    """
        A reset format specifier clears all formatting attributes.
        """
    self.assertAssembledEqually('\x02\x1fyay\x0freset', A.normal[A.bold[A.underline['yay']], 'reset'])
    self.assertAssembledEqually('\x0301yay\x0freset', A.normal[A.fg.black['yay'], 'reset'])
    self.assertAssembledEqually('\x0301,02yay\x0freset', A.normal[A.fg.black[A.bg.blue['yay']], 'reset'])