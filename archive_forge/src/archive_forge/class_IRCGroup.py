from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.internet.defer import succeed
from twisted.words.im import basesupport, interfaces, locals
from twisted.words.im.locals import ONLINE
from twisted.words.protocols import irc
@implementer(interfaces.IGroup)
class IRCGroup(basesupport.AbstractGroup):

    def imgroup_testAction(self):
        pass

    def imtarget_kick(self, target):
        if self.account.client is None:
            raise locals.OfflineError
        reason = 'for great justice!'
        self.account.client.sendLine(f'KICK #{self.name} {target.name} :{reason}')

    def setTopic(self, topic):
        if self.account.client is None:
            raise locals.OfflineError
        self.account.client.topic(self.name, topic)

    def sendGroupMessage(self, text, meta={}):
        if self.account.client is None:
            raise locals.OfflineError
        if meta and meta.get('style', None) == 'emote':
            self.account.client.ctcpMakeQuery(self.name, [('ACTION', text)])
            return succeed(text)
        for line in text.split('\n'):
            self.account.client.say(self.name, line)
        return succeed(text)

    def leave(self):
        if self.account.client is None:
            raise locals.OfflineError
        self.account.client.leave(self.name)
        self.account.client.getGroupConversation(self.name, 1)