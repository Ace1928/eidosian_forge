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
def test_removeHandler(self):
    """
        Test removal of a protocol handler.
        """
    handler = DummyXMPPHandler()
    handler.setHandlerParent(self.collection)
    handler.disownHandlerParent(self.collection)
    self.assertNotIn(handler, self.collection)
    self.assertIdentical(None, handler.parent)