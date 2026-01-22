import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class ClientTests(IRCTestCase):
    """
    Tests for the protocol-level behavior of IRCClient methods intended to
    be called by application code.
    """

    def setUp(self):
        """
        Create and connect a new L{IRCClient} to a new L{StringTransport}.
        """
        self.transport = StringTransport()
        self.protocol = IRCClient()
        self.protocol.performLogin = False
        self.protocol.makeConnection(self.transport)
        self.assertEqualBufferValue(self.transport.value(), '')
        self.addCleanup(self.transport.loseConnection)
        self.addCleanup(self.protocol.connectionLost, None)

    def getLastLine(self, transport):
        """
        Return the last IRC message in the transport buffer.
        """
        line = transport.value()
        if bytes != str and isinstance(line, bytes):
            line = line.decode('utf-8')
        return line.split('\r\n')[-2]

    def test_away(self):
        """
        L{IRCClient.away} sends an AWAY command with the specified message.
        """
        message = "Sorry, I'm not here."
        self.protocol.away(message)
        expected = [f'AWAY :{message}', '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_back(self):
        """
        L{IRCClient.back} sends an AWAY command with an empty message.
        """
        self.protocol.back()
        expected = ['AWAY :', '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_whois(self):
        """
        L{IRCClient.whois} sends a WHOIS message.
        """
        self.protocol.whois('alice')
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), ['WHOIS alice', ''])

    def test_whoisWithServer(self):
        """
        L{IRCClient.whois} sends a WHOIS message with a server name if a
        value is passed for the C{server} parameter.
        """
        self.protocol.whois('alice', 'example.org')
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), ['WHOIS example.org alice', ''])

    def test_register(self):
        """
        L{IRCClient.register} sends NICK and USER commands with the
        username, name, hostname, server name, and real name specified.
        """
        username = 'testuser'
        hostname = 'testhost'
        servername = 'testserver'
        self.protocol.realname = 'testname'
        self.protocol.password = None
        self.protocol.register(username, hostname, servername)
        expected = [f'NICK {username}', 'USER %s %s %s :%s' % (username, hostname, servername, self.protocol.realname), '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

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

    def test_registerWithTakenNick(self):
        """
        Verify that the client repeats the L{IRCClient.setNick} method with a
        new value when presented with an C{ERR_NICKNAMEINUSE} while trying to
        register.
        """
        username = 'testuser'
        hostname = 'testhost'
        servername = 'testserver'
        self.protocol.realname = 'testname'
        self.protocol.password = 'testpass'
        self.protocol.register(username, hostname, servername)
        self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertNotEqual(lastLine, f'NICK {username}')
        self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, 'NICK {}'.format(username + '__'))

    def test_overrideAlterCollidedNick(self):
        """
        L{IRCClient.alterCollidedNick} determines how a nickname is altered upon
        collision while a user is trying to change to that nickname.
        """
        nick = 'foo'
        self.protocol.alterCollidedNick = lambda nick: nick + '***'
        self.protocol.register(nick)
        self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, 'NICK {}'.format(nick + '***'))

    def test_nickChange(self):
        """
        When a NICK command is sent after signon, C{IRCClient.nickname} is set
        to the new nickname I{after} the server sends an acknowledgement.
        """
        oldnick = 'foo'
        newnick = 'bar'
        self.protocol.register(oldnick)
        self.protocol.irc_RPL_WELCOME('prefix', ['param'])
        self.protocol.setNick(newnick)
        self.assertEqual(self.protocol.nickname, oldnick)
        self.protocol.irc_NICK(f'{oldnick}!quux@qux', [newnick])
        self.assertEqual(self.protocol.nickname, newnick)

    def test_erroneousNick(self):
        """
        Trying to register an illegal nickname results in the default legal
        nickname being set, and trying to change a nickname to an illegal
        nickname results in the old nickname being kept.
        """
        badnick = 'foo'
        self.assertEqual(self.protocol._registered, False)
        self.protocol.register(badnick)
        self.protocol.irc_ERR_ERRONEUSNICKNAME('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, f'NICK {self.protocol.erroneousNickFallback}')
        self.protocol.irc_RPL_WELCOME('prefix', ['param'])
        self.assertEqual(self.protocol._registered, True)
        self.protocol.setNick(self.protocol.erroneousNickFallback)
        self.assertEqual(self.protocol.nickname, self.protocol.erroneousNickFallback)
        oldnick = self.protocol.nickname
        self.protocol.setNick(badnick)
        self.protocol.irc_ERR_ERRONEUSNICKNAME('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, f'NICK {badnick}')
        self.assertEqual(self.protocol.nickname, oldnick)

    def test_describe(self):
        """
        L{IRCClient.desrcibe} sends a CTCP ACTION message to the target
        specified.
        """
        target = 'foo'
        channel = '#bar'
        action = 'waves'
        self.protocol.describe(target, action)
        self.protocol.describe(channel, action)
        expected = [f'PRIVMSG {target} :\x01ACTION {action}\x01', f'PRIVMSG {channel} :\x01ACTION {action}\x01', '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_noticedDoesntPrivmsg(self):
        """
        The default implementation of L{IRCClient.noticed} doesn't invoke
        C{privmsg()}
        """

        def privmsg(user, channel, message):
            self.fail('privmsg() should not have been called')
        self.protocol.privmsg = privmsg
        self.protocol.irc_NOTICE('spam', ['#greasyspooncafe', "I don't want any spam!"])

    def test_ping(self):
        """
        L{IRCClient.ping}
        """
        self.protocol.ping('otheruser')
        self.assertTrue(self.transport.value().startswith(b'PRIVMSG otheruser :\x01PING'))
        self.transport.clear()
        self.protocol.ping('otheruser', 'are you there')
        self.assertEqual(self.transport.value(), b'PRIVMSG otheruser :\x01PING are you there\x01\r\n')
        self.transport.clear()
        self.protocol._pings = {}
        for pingNum in range(self.protocol._MAX_PINGRING + 3):
            self.protocol._pings['otheruser', str(pingNum)] = time.time() + pingNum
        self.assertEqual(len(self.protocol._pings), self.protocol._MAX_PINGRING + 3)
        self.protocol.ping('otheruser', 'I sent a lot of pings')
        self.assertEqual(len(self.protocol._pings), self.protocol._MAX_PINGRING)
        self.assertEqual(self.transport.value(), b'PRIVMSG otheruser :\x01PING I sent a lot of pings\x01\r\n')