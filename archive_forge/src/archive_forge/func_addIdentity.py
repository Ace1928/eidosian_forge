import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def addIdentity(self, blob, comment=b''):
    """
        Add a private key blob to the agent's collection of keys.
        """
    req = blob
    req += NS(comment)
    return self.sendRequest(AGENTC_ADD_IDENTITY, req)