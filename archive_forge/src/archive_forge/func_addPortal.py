from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def addPortal(self):
    r = TestRealm()
    p = portal.Portal(r)
    c = checkers.InMemoryUsernamePasswordDatabaseDontUse()
    c.addUser('userXname@127.0.0.1', 'passXword')
    p.registerChecker(c)
    self.proxy.portal = p