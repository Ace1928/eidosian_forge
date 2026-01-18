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
def test_notWantedRequired(self):
    """
        TLSRequired is raised when TLS is not wanted, but required by server.
        """
    tls = domish.Element(('urn:ietf:params:xml:ns:xmpp-tls', 'starttls'))
    tls.addElement('required')
    self.xmlstream.features = {(tls.uri, tls.name): tls}
    self.init.wanted = False
    d = self.init.start()
    self.assertEqual(1, len(self.output))
    self.assertFailure(d, xmlstream.TLSRequired)
    return d