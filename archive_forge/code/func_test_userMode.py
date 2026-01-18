import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_userMode(self):
    """
        A C{MODE} message whose target is our user (the nickname of our user,
        to be precise), as opposed to a channel, will be parsed according to
        the modes specified by L{IRCClient.getUserModeParams}.
        """
    target = self.client.nickname
    self._sendModeChange('+o', target=target)
    self._checkModeChange([(True, 'o', (None,))], target=target)

    def getUserModeParams():
        return ['Z', '']
    self.patch(self.client, 'getUserModeParams', getUserModeParams)
    self._sendModeChange('+Z', 'an_arg', target=target)
    self._checkModeChange([(True, 'Z', ('an_arg',))], target=target)