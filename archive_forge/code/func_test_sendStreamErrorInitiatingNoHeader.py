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
def test_sendStreamErrorInitiatingNoHeader(self):
    """
        Test sendStreamError on an initiating xmlstream without having sent a
        header.

        In this case, no header should be generated. Also, the error should
        not be sent out on the stream. Just closing the connection.
        """
    xs = self.xmlstream
    xs.initiating = True
    xs.transport.clear()
    xs.sendStreamError(error.StreamError('version-unsupported'))
    self.assertNot(xs._headerSent)
    self.assertEqual(b'', xs.transport.value())
    self.assertTrue(self.gotStreamEnd)