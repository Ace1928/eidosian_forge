from hashlib import sha1
from unittest import skipIf
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import client, error, jid, xmlstream
from twisted.words.protocols.jabber.sasl import SASLInitiatingInitializer
from twisted.words.xish import utility
class BindInitializerTests(InitiatingInitializerHarness, unittest.TestCase):
    """
    Tests for L{client.BindInitializer}.
    """

    def setUp(self):
        super().setUp()
        self.init = client.BindInitializer(self.xmlstream)
        self.authenticator.jid = jid.JID('user@example.com/resource')

    def testBasic(self):
        """
        Set up a stream, and act as if resource binding succeeds.
        """

        def onBind(iq):
            response = xmlstream.toResponse(iq, 'result')
            response.addElement((NS_BIND, 'bind'))
            response.bind.addElement('jid', content='user@example.com/other resource')
            self.pipe.source.send(response)

        def cb(result):
            self.assertEqual(jid.JID('user@example.com/other resource'), self.authenticator.jid)
        d1 = self.waitFor(IQ_BIND_SET, onBind)
        d2 = self.init.start()
        d2.addCallback(cb)
        return defer.gatherResults([d1, d2])

    def testFailure(self):
        """
        Set up a stream, and act as if resource binding fails.
        """

        def onBind(iq):
            response = error.StanzaError('conflict').toResponse(iq)
            self.pipe.source.send(response)
        d1 = self.waitFor(IQ_BIND_SET, onBind)
        d2 = self.init.start()
        self.assertFailure(d2, error.StanzaError)
        return defer.gatherResults([d1, d2])