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
def test_sendNotConnected(self):
    """
        Test send when there is no established XML stream.

        The data should be cached until an XML stream has been established and
        initialized.
        """
    factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
    sm = xmlstream.StreamManager(factory)
    handler = DummyXMPPHandler()
    sm.addHandler(handler)
    xs = factory.buildProtocol(None)
    xs.transport = proto_helpers.StringTransport()
    sm.send('<presence/>')
    self.assertEqual(b'', xs.transport.value())
    self.assertEqual('<presence/>', sm._packetQueue[0])
    xs.connectionMade()
    self.assertEqual(b'', xs.transport.value())
    self.assertEqual('<presence/>', sm._packetQueue[0])
    xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
    xs.dispatch(xs, '//event/stream/authd')
    self.assertEqual(b'<presence/>', xs.transport.value())
    self.assertFalse(sm._packetQueue)