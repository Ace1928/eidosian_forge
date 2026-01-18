import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_signDataCallbackErrorHandling(self):
    """
        Assert that L{SSHAgentClient.signData} raises a ConchError
        if we get a response from the server whose opcode doesn't match
        the protocol for data signing requests.
        """
    d = self.client.signData(self.rsaPublic.blob(), b'John Hancock')
    self.pump.flush()
    return self.assertFailure(d, ConchError)