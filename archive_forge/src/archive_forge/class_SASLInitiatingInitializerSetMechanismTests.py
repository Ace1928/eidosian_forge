from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
class SASLInitiatingInitializerSetMechanismTests(unittest.TestCase):
    """
    Test for L{sasl.SASLInitiatingInitializer.setMechanism}.
    """

    def setUp(self):
        self.output = []
        self.authenticator = xmlstream.Authenticator()
        self.xmlstream = xmlstream.XmlStream(self.authenticator)
        self.xmlstream.send = self.output.append
        self.xmlstream.connectionMade()
        self.xmlstream.dataReceived("<stream:stream xmlns='jabber:client' xmlns:stream='http://etherx.jabber.org/streams' from='example.com' id='12345' version='1.0'>")
        self.init = sasl.SASLInitiatingInitializer(self.xmlstream)

    def _setMechanism(self, name):
        """
        Set up the XML Stream to have a SASL feature with the given mechanism.
        """
        feature = domish.Element((NS_XMPP_SASL, 'mechanisms'))
        feature.addElement('mechanism', content=name)
        self.xmlstream.features[feature.uri, feature.name] = feature
        self.init.setMechanism()
        return self.init.mechanism.name

    def test_anonymous(self):
        """
        Test setting ANONYMOUS as the authentication mechanism.
        """
        self.authenticator.jid = jid.JID('example.com')
        self.authenticator.password = None
        name = 'ANONYMOUS'
        self.assertEqual(name, self._setMechanism(name))

    def test_plain(self):
        """
        Test setting PLAIN as the authentication mechanism.
        """
        self.authenticator.jid = jid.JID('test@example.com')
        self.authenticator.password = 'secret'
        name = 'PLAIN'
        self.assertEqual(name, self._setMechanism(name))

    def test_digest(self):
        """
        Test setting DIGEST-MD5 as the authentication mechanism.
        """
        self.authenticator.jid = jid.JID('test@example.com')
        self.authenticator.password = 'secret'
        name = 'DIGEST-MD5'
        self.assertEqual(name, self._setMechanism(name))

    def test_notAcceptable(self):
        """
        Test using an unacceptable SASL authentication mechanism.
        """
        self.authenticator.jid = jid.JID('test@example.com')
        self.authenticator.password = 'secret'
        self.assertRaises(sasl.SASLNoAcceptableMechanism, self._setMechanism, 'SOMETHING_UNACCEPTABLE')

    def test_notAcceptableWithoutUser(self):
        """
        Test using an unacceptable SASL authentication mechanism with no JID.
        """
        self.authenticator.jid = jid.JID('example.com')
        self.authenticator.password = 'secret'
        self.assertRaises(sasl.SASLNoAcceptableMechanism, self._setMechanism, 'SOMETHING_UNACCEPTABLE')