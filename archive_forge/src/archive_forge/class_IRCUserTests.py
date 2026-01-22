from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
class IRCUserTests(IRCTestCase):
    """
    Isolated tests for L{IRCUser}
    """

    def setUp(self):
        """
        Sets up a Realm, Portal, Factory, IRCUser, Transport, and Connection
        for our tests.
        """
        self.realm = InMemoryWordsRealm('example.com')
        self.checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.portal = portal.Portal(self.realm, [self.checker])
        self.checker.addUser('john', 'pass')
        self.factory = IRCFactory(self.realm, self.portal)
        self.ircUser = self.factory.buildProtocol(None)
        self.stringTransport = proto_helpers.StringTransport()
        self.ircUser.makeConnection(self.stringTransport)

    def test_sendMessage(self):
        """
        Sending a message to a user after they have sent NICK, but before they
        have authenticated, results in a message from "example.com".
        """
        self.ircUser.irc_NICK('', ['mynick'])
        self.stringTransport.clear()
        self.ircUser.sendMessage('foo')
        self.assertEqualBufferValue(self.stringTransport.value(), ':example.com foo mynick\r\n')

    def test_utf8Messages(self):
        """
        When a UTF8 message is sent with sendMessage and the current IRCUser
        has a UTF8 nick and is set to UTF8 encoding, the message will be
        written to the transport.
        """
        expectedResult = ':example.com тест ник\r\n'.encode()
        self.ircUser.irc_NICK('', ['ник'.encode()])
        self.stringTransport.clear()
        self.ircUser.sendMessage('тест'.encode())
        self.assertEqualBufferValue(self.stringTransport.value(), expectedResult)

    def test_invalidEncodingNick(self):
        """
        A NICK command sent with a nickname that cannot be decoded with the
        current IRCUser's encoding results in a PRIVMSG from NickServ
        indicating that the nickname could not be decoded.
        """
        self.ircUser.irc_NICK('', [b'\xd4\xc5\xd3\xd4'])
        self.assertRaises(UnicodeError)

    def response(self):
        """
        Grabs our responses and then clears the transport
        """
        response = self.ircUser.transport.value()
        self.ircUser.transport.clear()
        if bytes != str and isinstance(response, bytes):
            response = response.decode('utf-8')
        response = response.splitlines()
        return [irc.parsemsg(r) for r in response]

    def scanResponse(self, response, messageType):
        """
        Gets messages out of a response

        @param response: The parsed IRC messages of the response, as returned
        by L{IRCUserTests.response}

        @param messageType: The string type of the desired messages.

        @return: An iterator which yields 2-tuples of C{(index, ircMessage)}
        """
        for n, message in enumerate(response):
            if message[1] == messageType:
                yield (n, message)

    def test_sendNickSendsGreeting(self):
        """
        Receiving NICK without authenticating sends the MOTD Start and MOTD End
        messages, which is required by certain popular IRC clients (such as
        Pidgin) before a connection is considered to be fully established.
        """
        self.ircUser.irc_NICK('', ['mynick'])
        response = self.response()
        start = list(self.scanResponse(response, irc.RPL_MOTDSTART))
        end = list(self.scanResponse(response, irc.RPL_ENDOFMOTD))
        self.assertEqual(start, [(0, ('example.com', '375', ['mynick', '- example.com Message of the Day - ']))])
        self.assertEqual(end, [(1, ('example.com', '376', ['mynick', 'End of /MOTD command.']))])

    def test_fullLogin(self):
        """
        Receiving USER, PASS, NICK will log in the user, and transmit the
        appropriate response messages.
        """
        self.ircUser.irc_USER('', ['john doe'])
        self.ircUser.irc_PASS('', ['pass'])
        self.ircUser.irc_NICK('', ['john'])
        version = 'Your host is example.com, running version {}'.format(self.factory._serverInfo['serviceVersion'])
        creation = 'This server was created on {}'.format(self.factory._serverInfo['creationDate'])
        self.assertEqual(self.response(), [('example.com', '375', ['john', '- example.com Message of the Day - ']), ('example.com', '376', ['john', 'End of /MOTD command.']), ('example.com', '001', ['john', 'connected to Twisted IRC']), ('example.com', '002', ['john', version]), ('example.com', '003', ['john', creation]), ('example.com', '004', ['john', 'example.com', self.factory._serverInfo['serviceVersion'], 'w', 'n'])])

    def test_PART(self):
        """
        irc_PART
        """
        self.ircUser.irc_NICK('testuser', ['mynick'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.RPL_MOTDSTART)
        self.ircUser.irc_JOIN('testuser', ['somechannel'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.ERR_NOSUCHCHANNEL)
        self.ircUser.irc_PART('testuser', [b'somechannel', b'booga'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.ERR_NOTONCHANNEL)
        self.ircUser.irc_PART('testuser', ['somechannel', 'booga'])
        response = self.response()
        self.ircUser.transport.clear()
        self.assertEqual(response[0][1], irc.ERR_NOTONCHANNEL)

    def test_NAMES(self):
        """
        irc_NAMES
        """
        self.ircUser.irc_NICK('', ['testuser'])
        self.ircUser.irc_JOIN('', ['somechannel'])
        self.ircUser.transport.clear()
        self.ircUser.irc_NAMES('', ['somechannel'])
        response = self.response()
        self.assertEqual(response[0][1], irc.RPL_ENDOFNAMES)