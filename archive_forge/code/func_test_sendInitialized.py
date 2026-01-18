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
def test_sendInitialized(self):
    """
        Test send when the stream has been initialized.

        The data should be sent directly over the XML stream.
        """
    factory = xmlstream.XmlStreamFactory(xmlstream.Authenticator())
    sm = xmlstream.StreamManager(factory)
    xs = factory.buildProtocol(None)
    xs.transport = proto_helpers.StringTransport()
    xs.connectionMade()
    xs.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345'>")
    xs.dispatch(xs, '//event/stream/authd')
    sm.send('<presence/>')
    self.assertEqual(b'<presence/>', xs.transport.value())