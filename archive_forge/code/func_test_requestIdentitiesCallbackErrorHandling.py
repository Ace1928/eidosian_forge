import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_requestIdentitiesCallbackErrorHandling(self):
    """
        Assert that L{SSHAgentClient.requestIdentities} raises a ConchError
        if we get a response from the server whose opcode doesn't match
        the protocol for identity requests.
        """
    d = self.client.requestIdentities()
    self.pump.flush()
    return self.assertFailure(d, ConchError)