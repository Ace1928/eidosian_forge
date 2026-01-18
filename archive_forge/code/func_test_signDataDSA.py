import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_signDataDSA(self):
    """
        Sign data with a DSA private key and then verify it with the public
        key.
        """
    d = self.client.signData(self.dsaPublic.blob(), b'John Hancock')
    self.pump.flush()

    def _check(sig):
        self.assertTrue(self.dsaPublic.verify(sig, b'John Hancock'))
    return d.addCallback(_check)