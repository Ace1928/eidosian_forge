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
def test_sendNotInitialized(self):
    """
        Test send when the stream is connected but not yet initialized.

        The data should be cached until the XML stream has been initialized.
        """
    factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
    sm = xmlstream.StreamManager(factory)
    xs = factory.buildProtocol(None)
    xs.transport = proto_helpers.StringTransport()
    xs.connectionMade()
    xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
    sm.send('<presence/>')
    self.assertEqual(b'', xs.transport.value())
    self.assertEqual('<presence/>', sm._packetQueue[0])