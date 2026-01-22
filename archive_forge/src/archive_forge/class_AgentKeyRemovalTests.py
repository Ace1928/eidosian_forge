import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
class AgentKeyRemovalTests(AgentTestBase):
    """
    Test support for removing keys in a remote server.
    """

    def setUp(self):
        AgentTestBase.setUp(self)
        self.server.factory.keys[self.dsaPrivate.blob()] = (self.dsaPrivate, b'a comment')
        self.server.factory.keys[self.rsaPrivate.blob()] = (self.rsaPrivate, b'another comment')

    def test_removeRSAIdentity(self):
        """
        Assert that we can remove an RSA identity.
        """
        d = self.client.removeIdentity(self.rsaPrivate.blob())
        self.pump.flush()

        def _check(ignored):
            self.assertEqual(1, len(self.server.factory.keys))
            self.assertIn(self.dsaPrivate.blob(), self.server.factory.keys)
            self.assertNotIn(self.rsaPrivate.blob(), self.server.factory.keys)
        return d.addCallback(_check)

    def test_removeDSAIdentity(self):
        """
        Assert that we can remove a DSA identity.
        """
        d = self.client.removeIdentity(self.dsaPrivate.blob())
        self.pump.flush()

        def _check(ignored):
            self.assertEqual(1, len(self.server.factory.keys))
            self.assertIn(self.rsaPrivate.blob(), self.server.factory.keys)
        return d.addCallback(_check)

    def test_removeAllIdentities(self):
        """
        Assert that we can remove all identities.
        """
        d = self.client.removeAllIdentities()
        self.pump.flush()

        def _check(ignored):
            self.assertEqual(0, len(self.server.factory.keys))
        return d.addCallback(_check)