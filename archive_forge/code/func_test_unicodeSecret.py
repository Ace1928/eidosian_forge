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
def test_unicodeSecret(self):
    """
        The concatenated sid and password must be encoded to UTF-8 before hashing.
        """
    hash = xmlstream.hashPassword('12345', 'secrét')
    self.assertEqual('659bf88d8f8e179081f7f3b4a8e7d224652d2853', hash)