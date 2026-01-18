from twisted.internet.testing import StringTransport
from twisted.words.im.basechat import ChatUI, Conversation, GroupConversation
from twisted.words.im.ircsupport import IRCAccount, IRCProto
from twisted.words.im.locals import OfflineError
from twisted.words.test.test_irc import IRCTestCase
def test_privmsg(self) -> None:
    """
        PRIVMSG sends a private message to a user or channel.
        """
    self.proto.makeConnection(self.transport)
    self.proto.dataReceived(':alice1 PRIVMSG t2 test_message_1\r\n')
    conversation = self.proto.chat.getConversation(self.proto.getPerson('alice1'))
    self.assertEqual(conversation.message, 'test_message_1')
    self.proto.dataReceived(':alice1 PRIVMSG #group1 test_message_2\r\n')
    groupConversation = self.proto.getGroupConversation('group1')
    self.assertEqual(groupConversation.text, 'test_message_2')
    self.proto.setNick('alice')
    self.proto.dataReceived(':alice PRIVMSG #foo test_message_3\r\n')
    groupConversation = self.proto.getGroupConversation('foo')
    self.assertFalse(hasattr(groupConversation, 'text'))
    conversation = self.proto.chat.getConversation(self.proto.getPerson('alice'))
    self.assertFalse(hasattr(conversation, 'message'))