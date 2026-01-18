from hashlib import sha1
from zope.interface.verify import verifyObject
from twisted.python import failure
from twisted.trial import unittest
from twisted.words.protocols.jabber import component, ijabber, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish
from twisted.words.xish.utility import XmlPipe
def test_route(self):
    """
        Test routing of a message.
        """
    component1 = XmlPipe()
    component2 = XmlPipe()
    router = component.Router()
    router.addRoute('component1.example.org', component1.sink)
    router.addRoute('component2.example.org', component2.sink)
    outgoing = []
    component2.source.addObserver('/*', lambda element: outgoing.append(element))
    stanza = domish.Element((None, 'presence'))
    stanza['from'] = 'component1.example.org'
    stanza['to'] = 'component2.example.org'
    component1.source.send(stanza)
    self.assertEqual([stanza], outgoing)