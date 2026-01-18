from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_onHandshake(self):
    """
        Receiving a handshake matching the secret authenticates the stream.
        """
    authd = []

    def authenticated(xs):
        authd.append(xs)
    xs = self.xmlstream
    xs.addOnetimeObserver(xmlstream.STREAM_AUTHD_EVENT, authenticated)
    xs.sid = '1234'
    theHash = '32532c0f7dbf1253c095b18b18e36d38d94c1256'
    xs.authenticator.onHandshake(theHash)
    self.assertEqual('<handshake/>', self.output[-1])
    self.assertEqual(1, len(authd))