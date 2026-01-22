from zope.interface import implementer
from twisted.internet import defer, error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from twisted.words.im import basesupport, interfaces
from twisted.words.im.locals import AWAY, OFFLINE, ONLINE
def _cbIdent(self, ident, chatui):
    if not ident:
        print('falsely identified.')
        return self._ebConnected(Failure(Exception('username or password incorrect')))
    print('Identified!')
    dl = []
    for handlerClass, sname, pname in self.services:
        d = defer.Deferred()
        dl.append(d)
        handler = handlerClass(self, sname, pname, chatui, d)
        ident.callRemote('attach', sname, pname, handler).addCallback(handler.connected)
    return defer.DeferredList(dl)