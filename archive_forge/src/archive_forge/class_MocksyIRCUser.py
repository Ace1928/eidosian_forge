from twisted.cred import checkers, portal
from twisted.test import proto_helpers
from twisted.words.protocols import irc
from twisted.words.service import InMemoryWordsRealm, IRCFactory, IRCUser
from twisted.words.test.test_irc import IRCTestCase
class MocksyIRCUser(IRCUser):

    def __init__(self):
        self.realm = InMemoryWordsRealm('example.com')
        self.mockedCodes = []

    def sendMessage(self, code, *_, **__):
        self.mockedCodes.append(code)