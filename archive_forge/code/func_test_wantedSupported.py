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
@skipIf(*skipWhenNoSSL)
def test_wantedSupported(self):
    """
        When TLS is wanted and SSL available, StartTLS is initiated.
        """
    self.xmlstream.transport = proto_helpers.StringTransport()
    self.xmlstream.transport.startTLS = lambda ctx: self.done.append('TLS')
    self.xmlstream.reset = lambda: self.done.append('reset')
    self.xmlstream.sendHeader = lambda: self.done.append('header')
    d = self.init.start()
    d.addCallback(self.assertEqual, xmlstream.Reset)
    self.assertEqual(2, len(self.output))
    starttls = self.output[1]
    self.assertEqual('starttls', starttls.name)
    self.assertEqual(NS_XMPP_TLS, starttls.uri)
    self.xmlstream.dataReceived("<proceed xmlns='%s'/>" % NS_XMPP_TLS)
    self.assertEqual(['TLS', 'reset', 'header'], self.done)
    return d