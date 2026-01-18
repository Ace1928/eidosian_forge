from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_onHandshakeWrongHash(self):
    """
        Receiving a bad handshake should yield a stream error.
        """
    streamErrors = []
    authd = []

    def authenticated(xs):
        authd.append(xs)
    xs = self.xmlstream
    xs.addOnetimeObserver(xmlstream.STREAM_AUTHD_EVENT, authenticated)
    xs.sendStreamError = streamErrors.append
    xs.sid = '1234'
    theHash = '1234'
    xs.authenticator.onHandshake(theHash)
    self.assertEqual('not-authorized', streamErrors[-1].condition)
    self.assertEqual(0, len(authd))