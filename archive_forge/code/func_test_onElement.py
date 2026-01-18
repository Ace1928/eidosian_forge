from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_onElement(self):
    """
        We expect a handshake element with a hash.
        """
    handshakes = []
    xs = self.xmlstream
    xs.authenticator.onHandshake = handshakes.append
    handshake = domish.Element(('jabber:component:accept', 'handshake'))
    handshake.addContent('1234')
    xs.authenticator.onElement(handshake)
    self.assertEqual('1234', handshakes[-1])