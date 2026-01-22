from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def _cbHandshake(self, _):
    self.xmlstream.thisEntity = self.xmlstream.otherEntity
    self._deferred.callback(None)