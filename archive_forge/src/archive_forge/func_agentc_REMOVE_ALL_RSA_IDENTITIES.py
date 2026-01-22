import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_REMOVE_ALL_RSA_IDENTITIES(self, data):
    """
        v1 message for removing all RSA1 keys; superseded by
        agentc_REMOVE_ALL_IDENTITIES, which handles different key types.
        """
    self.sendResponse(AGENT_SUCCESS, b'')