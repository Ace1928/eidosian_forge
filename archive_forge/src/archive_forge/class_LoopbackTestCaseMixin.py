from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
class LoopbackTestCaseMixin:

    def testRegularFunction(self):
        s = SimpleProtocol()
        c = SimpleProtocol()

        def sendALine(result):
            s.sendLine(b'THIS IS LINE ONE!')
            s.transport.loseConnection()
        s.conn.addCallback(sendALine)

        def check(ignored):
            self.assertEqual(c.lines, [b'THIS IS LINE ONE!'])
            self.assertEqual(len(s.connLost), 1)
            self.assertEqual(len(c.connLost), 1)
        d = defer.maybeDeferred(self.loopbackFunc, s, c)
        d.addCallback(check)
        return d

    def testSneakyHiddenDoom(self):
        s = DoomProtocol()
        c = DoomProtocol()

        def sendALine(result):
            s.sendLine(b'DOOM LINE')
        s.conn.addCallback(sendALine)

        def check(ignored):
            self.assertEqual(s.lines, [b'Hello 1', b'Hello 2', b'Hello 3'])
            self.assertEqual(c.lines, [b'DOOM LINE', b'Hello 1', b'Hello 2', b'Hello 3'])
            self.assertEqual(len(s.connLost), 1)
            self.assertEqual(len(c.connLost), 1)
        d = defer.maybeDeferred(self.loopbackFunc, s, c)
        d.addCallback(check)
        return d