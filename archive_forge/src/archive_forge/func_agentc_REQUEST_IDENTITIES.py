import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def agentc_REQUEST_IDENTITIES(self, data):
    """
        Return all of the identities that have been added to the server
        """
    assert data == b''
    numKeys = len(self.factory.keys)
    resp = []
    resp.append(struct.pack('!L', numKeys))
    for key, comment in self.factory.keys.values():
        resp.append(NS(key.blob()))
        resp.append(NS(comment))
    self.sendResponse(AGENT_IDENTITIES_ANSWER, b''.join(resp))