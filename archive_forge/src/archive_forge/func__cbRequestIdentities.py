import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
def _cbRequestIdentities(self, data):
    """
        Unpack a collection of identities into a list of tuples comprised of
        public key blobs and comments.
        """
    if ord(data[0:1]) != AGENT_IDENTITIES_ANSWER:
        raise ConchError('unexpected response: %i' % ord(data[0:1]))
    numKeys = struct.unpack('!L', data[1:5])[0]
    result = []
    data = data[5:]
    for i in range(numKeys):
        blob, data = getNS(data)
        comment, data = getNS(data)
        result.append((blob, comment))
    return result