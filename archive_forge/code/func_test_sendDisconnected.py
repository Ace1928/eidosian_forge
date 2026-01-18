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
def test_sendDisconnected(self):
    """
        Test send after XML stream disconnection.

        The data should be cached until a new XML stream has been established
        and initialized.
        """
    factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
    sm = xmlstream.StreamManager(factory)
    handler = DummyXMPPHandler()
    sm.addHandler(handler)
    xs = factory.buildProtocol(None)
    xs.connectionMade()
    xs.transport = proto_helpers.StringTransport()
    xs.connectionLost(None)
    sm.send('<presence/>')
    self.assertEqual(b'', xs.transport.value())
    self.assertEqual('<presence/>', sm._packetQueue[0])