from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, task
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IProtocolFactory
from twisted.python import failure
from twisted.test import proto_helpers
from twisted.trial import unittest
from twisted.words.protocols.jabber import error, ijabber, jid, xmlstream
from twisted.words.test.test_xmlstream import GenericXmlStreamFactoryTestsMixin
from twisted.words.xish import domish
def test_toResponseNoTo(self):
    """
        Test that a response is generated from a stanza without a to address.
        """
    stanza = domish.Element(('jabber:client', 'iq'))
    stanza['type'] = 'get'
    stanza['from'] = 'user2@example.com/resource'
    response = xmlstream.toResponse(stanza)
    self.assertFalse(response.hasAttribute('from'))
    self.assertEqual(response['to'], 'user2@example.com/resource')