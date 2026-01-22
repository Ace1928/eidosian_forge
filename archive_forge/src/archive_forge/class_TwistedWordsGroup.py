from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
@implementer(interfaces.IGroup)
class TwistedWordsGroup(basesupport.AbstractGroup):

    def __init__(self, name, wordsClient):
        basesupport.AbstractGroup.__init__(self, name, wordsClient)
        self.joined = 0

    def sendGroupMessage(self, text, metadata=None):
        """Return a deferred."""
        if metadata:
            d = self.account.client.perspective.callRemote('groupMessage', self.name, text, metadata)
            d.addErrback(self.metadataFailed, '* ' + text)
            return d
        else:
            return self.account.client.perspective.callRemote('groupMessage', self.name, text)

    def setTopic(self, text):
        self.account.client.perspective.callRemote('setGroupMetadata', {'topic': text, 'topic_author': self.client.name}, self.name)

    def metadataFailed(self, result, text):
        print('result:', result, 'text:', text)
        return self.account.client.perspective.callRemote('groupMessage', self.name, text)

    def joining(self):
        self.joined = 1

    def leaving(self):
        self.joined = 0

    def leave(self):
        return self.account.client.perspective.callRemote('leaveGroup', self.name)