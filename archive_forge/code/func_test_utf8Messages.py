from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
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