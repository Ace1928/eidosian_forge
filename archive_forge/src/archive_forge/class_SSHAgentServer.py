import struct
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.ssh import keys
from twisted.conch.ssh.common import NS, getMP, getNS
from twisted.internet import defer, protocol
class SSHAgentServer(protocol.Protocol):
    """
    The server side of the SSH agent protocol.  This is equivalent to
    ssh-agent(1) and can be used with either ssh-add(1) or the SSHAgentClient
    protocol, also in this package.
    """

    def __init__(self):
        self.buf = b''

    def dataReceived(self, data):
        self.buf += data
        while 1:
            if len(self.buf) <= 4:
                return
            packLen = struct.unpack('!L', self.buf[:4])[0]
            if len(self.buf) < 4 + packLen:
                return
            packet, self.buf = (self.buf[4:4 + packLen], self.buf[4 + packLen:])
            reqType = ord(packet[0:1])
            reqName = messages.get(reqType, None)
            if not reqName:
                self.sendResponse(AGENT_FAILURE, b'')
            else:
                f = getattr(self, 'agentc_%s' % reqName)
                if getattr(self.factory, 'keys', None) is None:
                    self.sendResponse(AGENT_FAILURE, b'')
                    raise MissingKeyStoreError()
                f(packet[1:])

    def sendResponse(self, reqType, data):
        pack = struct.pack('!LB', len(data) + 1, reqType) + data
        self.transport.write(pack)

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

    def agentc_SIGN_REQUEST(self, data):
        """
        Data is a structure with a reference to an already added key object and
        some data that the clients wants signed with that key.  If the key
        object wasn't loaded, return AGENT_FAILURE, else return the signature.
        """
        blob, data = getNS(data)
        if blob not in self.factory.keys:
            return self.sendResponse(AGENT_FAILURE, b'')
        signData, data = getNS(data)
        assert data == b'\x00\x00\x00\x00'
        self.sendResponse(AGENT_SIGN_RESPONSE, NS(self.factory.keys[blob][0].sign(signData)))

    def agentc_ADD_IDENTITY(self, data):
        """
        Adds a private key to the agent's collection of identities.  On
        subsequent interactions, the private key can be accessed using only the
        corresponding public key.
        """
        keyType, rest = getNS(data)
        if keyType == b'ssh-rsa':
            nmp = 6
        elif keyType == b'ssh-dss':
            nmp = 5
        else:
            raise keys.BadKeyError('unknown blob type: %s' % keyType)
        rest = getMP(rest, nmp)[-1]
        comment, rest = getNS(rest)
        k = keys.Key.fromString(data, type='private_blob')
        self.factory.keys[k.blob()] = (k, comment)
        self.sendResponse(AGENT_SUCCESS, b'')

    def agentc_REMOVE_IDENTITY(self, data):
        """
        Remove a specific key from the agent's collection of identities.
        """
        blob, _ = getNS(data)
        k = keys.Key.fromString(blob, type='blob')
        del self.factory.keys[k.blob()]
        self.sendResponse(AGENT_SUCCESS, b'')

    def agentc_REMOVE_ALL_IDENTITIES(self, data):
        """
        Remove all keys from the agent's collection of identities.
        """
        assert data == b''
        self.factory.keys = {}
        self.sendResponse(AGENT_SUCCESS, b'')

    def agentc_REQUEST_RSA_IDENTITIES(self, data):
        """
        v1 message for listing RSA1 keys; superseded by
        agentc_REQUEST_IDENTITIES, which handles different key types.
        """
        self.sendResponse(AGENT_RSA_IDENTITIES_ANSWER, struct.pack('!L', 0))

    def agentc_REMOVE_RSA_IDENTITY(self, data):
        """
        v1 message for removing RSA1 keys; superseded by
        agentc_REMOVE_IDENTITY, which handles different key types.
        """
        self.sendResponse(AGENT_SUCCESS, b'')

    def agentc_REMOVE_ALL_RSA_IDENTITIES(self, data):
        """
        v1 message for removing all RSA1 keys; superseded by
        agentc_REMOVE_ALL_IDENTITIES, which handles different key types.
        """
        self.sendResponse(AGENT_SUCCESS, b'')