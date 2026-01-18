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
def test_wantedNotSupportedRequired(self):
    """
        TLSNotSupported is raised when TLS is required but not available.
        """
    xmlstream.ssl = None
    self.init.required = True
    d = self.init.start()
    self.assertFailure(d, xmlstream.TLSNotSupported)
    self.assertEqual(1, len(self.output))
    return d