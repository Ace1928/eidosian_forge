import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def test_registerWithPassword(self):
    """
        If the C{password} attribute of L{IRCClient} is not L{None}, the
        C{register} method also sends a PASS command with it as the
        argument.
        """
    username = 'testuser'
    hostname = 'testhost'
    servername = 'testserver'
    self.protocol.realname = 'testname'
    self.protocol.password = 'testpass'
    self.protocol.register(username, hostname, servername)
    expected = [f'PASS {self.protocol.password}', f'NICK {username}', 'USER %s %s %s :%s' % (username, hostname, servername, self.protocol.realname), '']
    self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)