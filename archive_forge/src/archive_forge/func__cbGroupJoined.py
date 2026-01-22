from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def _cbGroupJoined(self, result, name):
    groupConv = self.chat.getGroupConversation(self.getGroup(name))
    groupConv.showGroupMessage('sys', 'you joined')
    self.perspective.callRemote('getGroupMembers', name)