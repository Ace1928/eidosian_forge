from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
def _connected(self, xs):
    self.xmlstream = xs
    for c in self:
        if ijabber.IService.providedBy(c):
            c.transportConnected(xs)