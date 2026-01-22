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
class XmlStreamServerFactoryTests(GenericXmlStreamFactoryTestsMixin):
    """
    Tests for L{xmlstream.XmlStreamServerFactory}.
    """

    def setUp(self):
        """
        Set up a server factory with an authenticator factory function.
        """

        class TestAuthenticator:

            def __init__(self):
                self.xmlstreams = []

            def associateWithStream(self, xs):
                self.xmlstreams.append(xs)

        def authenticatorFactory():
            return TestAuthenticator()
        self.factory = xmlstream.XmlStreamServerFactory(authenticatorFactory)

    def test_interface(self):
        """
        L{XmlStreamServerFactory} is a L{Factory}.
        """
        verifyObject(IProtocolFactory, self.factory)

    def test_buildProtocolAuthenticatorInstantiation(self):
        """
        The authenticator factory should be used to instantiate the
        authenticator and pass it to the protocol.

        The default protocol, L{XmlStream} stores the authenticator it is
        passed, and calls its C{associateWithStream} method. so we use that to
        check whether our authenticator factory is used and the protocol
        instance gets an authenticator.
        """
        xs = self.factory.buildProtocol(None)
        self.assertEqual([xs], xs.authenticator.xmlstreams)

    def test_buildProtocolXmlStream(self):
        """
        The protocol factory creates Jabber XML Stream protocols by default.
        """
        xs = self.factory.buildProtocol(None)
        self.assertIsInstance(xs, xmlstream.XmlStream)

    def test_buildProtocolTwice(self):
        """
        Subsequent calls to buildProtocol should result in different instances
        of the protocol, as well as their authenticators.
        """
        xs1 = self.factory.buildProtocol(None)
        xs2 = self.factory.buildProtocol(None)
        self.assertNotIdentical(xs1, xs2)
        self.assertNotIdentical(xs1.authenticator, xs2.authenticator)