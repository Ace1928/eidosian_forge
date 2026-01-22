from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
class JabberServiceManagerTests(unittest.TestCase):

    def testSM(self):
        sm = component.ServiceManager('foo', 'password')
        svc = JabberServiceHarness()
        svc.setServiceParent(sm)
        wlist = []
        xs = sm.getFactory().buildProtocol(None)
        xs.transport = self
        xs.transport.write = wlist.append
        xs.connectionMade()
        self.assertEqual(True, svc.transportConnectedFlag)
        xs.dispatch(xs, xmlstream.STREAM_AUTHD_EVENT)
        self.assertEqual(True, svc.componentConnectedFlag)
        xs.connectionLost(None)
        self.assertEqual(True, svc.componentDisconnectedFlag)