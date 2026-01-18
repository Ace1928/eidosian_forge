import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_oneModeParameter(self):
    """
        Parameters are passed to L{IRCClient.modeChanged} for modes that take
        parameters.
        """
    self._sendModeChange('+o', 'a_user')
    self._checkModeChange([(True, 'o', ('a_user',))])
    self._sendModeChange('-o', 'a_user')
    self._checkModeChange([(False, 'o', ('a_user',))])