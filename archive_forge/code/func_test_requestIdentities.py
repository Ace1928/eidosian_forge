import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_requestIdentities(self):
    """
        Assert that we get all of the keys/comments that we add when we issue a
        request for all identities.
        """
    d = self.client.requestIdentities()
    self.pump.flush()

    def _check(keyt):
        expected = {}
        expected[self.dsaPublic.blob()] = b'a comment'
        expected[self.rsaPublic.blob()] = b'another comment'
        received = {}
        for k in keyt:
            received[keys.Key.fromString(k[0], type='blob').blob()] = k[1]
        self.assertEqual(expected, received)
    return d.addCallback(_check)