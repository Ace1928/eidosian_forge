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
class IQTests(unittest.TestCase):
    """
    Tests both IQ and the associated IIQResponseTracker callback.
    """

    def setUp(self):
        authenticator = xmlstream.ConnectAuthenticator('otherhost')
        authenticator.namespace = 'testns'
        self.xmlstream = xmlstream.XmlStream(authenticator)
        self.clock = task.Clock()
        self.xmlstream._callLater = self.clock.callLater
        self.xmlstream.makeConnection(proto_helpers.StringTransport())
        self.xmlstream.dataReceived("<stream:stream xmlns:stream='http://etherx.jabber.org/streams' xmlns='testns' from='otherhost' version='1.0'>")
        self.iq = xmlstream.IQ(self.xmlstream, 'get')

    def testBasic(self):
        self.assertEqual(self.iq['type'], 'get')
        self.assertTrue(self.iq['id'])

    def testSend(self):
        self.xmlstream.transport.clear()
        self.iq.send()
        idBytes = self.iq['id'].encode('utf-8')
        self.assertIn(self.xmlstream.transport.value(), [b"<iq type='get' id='" + idBytes + b"'/>", b"<iq id='" + idBytes + b"' type='get'/>"])

    def testResultResponse(self):

        def cb(result):
            self.assertEqual(result['type'], 'result')
        d = self.iq.send()
        d.addCallback(cb)
        xs = self.xmlstream
        xs.dataReceived("<iq type='result' id='%s'/>" % self.iq['id'])
        return d

    def testErrorResponse(self):
        d = self.iq.send()
        self.assertFailure(d, error.StanzaError)
        xs = self.xmlstream
        xs.dataReceived("<iq type='error' id='%s'/>" % self.iq['id'])
        return d

    def testNonTrackedResponse(self):
        """
        Test that untracked iq responses don't trigger any action.

        Untracked means that the id of the incoming response iq is not
        in the stream's C{iqDeferreds} dictionary.
        """
        xs = self.xmlstream
        xmlstream.upgradeWithIQResponseTracker(xs)
        self.assertFalse(xs.iqDeferreds)

        def cb(iq):
            self.assertFalse(getattr(iq, 'handled', False))
        xs.addObserver('/iq', cb, -1)
        xs.dataReceived("<iq type='result' id='test'/>")

    def testCleanup(self):
        """
        Test if the deferred associated with an iq request is removed
        from the list kept in the L{XmlStream} object after it has
        been fired.
        """
        d = self.iq.send()
        xs = self.xmlstream
        xs.dataReceived("<iq type='result' id='%s'/>" % self.iq['id'])
        self.assertNotIn(self.iq['id'], xs.iqDeferreds)
        return d

    def testDisconnectCleanup(self):
        """
        Test if deferreds for iq's that haven't yet received a response
        have their errback called on stream disconnect.
        """
        d = self.iq.send()
        xs = self.xmlstream
        xs.connectionLost('Closed by peer')
        self.assertFailure(d, ConnectionLost)
        return d

    def testNoModifyingDict(self):
        """
        Test to make sure the errbacks cannot cause the iteration of the
        iqDeferreds to blow up in our face.
        """

        def eb(failure):
            d = xmlstream.IQ(self.xmlstream).send()
            d.addErrback(eb)
        d = self.iq.send()
        d.addErrback(eb)
        self.xmlstream.connectionLost('Closed by peer')
        return d

    def testRequestTimingOut(self):
        """
        Test that an iq request with a defined timeout times out.
        """
        self.iq.timeout = 60
        d = self.iq.send()
        self.assertFailure(d, xmlstream.TimeoutError)
        self.clock.pump([1, 60])
        self.assertFalse(self.clock.calls)
        self.assertFalse(self.xmlstream.iqDeferreds)
        return d

    def testRequestNotTimingOut(self):
        """
        Test that an iq request with a defined timeout does not time out
        when a response was received before the timeout period elapsed.
        """
        self.iq.timeout = 60
        d = self.iq.send()
        self.clock.callLater(1, self.xmlstream.dataReceived, "<iq type='result' id='%s'/>" % self.iq['id'])
        self.clock.pump([1, 1])
        self.assertFalse(self.clock.calls)
        return d

    def testDisconnectTimeoutCancellation(self):
        """
        Test if timeouts for iq's that haven't yet received a response
        are cancelled on stream disconnect.
        """
        self.iq.timeout = 60
        d = self.iq.send()
        xs = self.xmlstream
        xs.connectionLost('Closed by peer')
        self.assertFailure(d, ConnectionLost)
        self.assertFalse(self.clock.calls)
        return d