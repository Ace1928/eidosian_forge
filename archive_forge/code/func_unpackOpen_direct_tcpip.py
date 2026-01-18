import struct
from twisted.conch.ssh import channel, common
from twisted.internet import protocol, reactor
from twisted.internet.endpoints import HostnameEndpoint, connectProtocol
def unpackOpen_direct_tcpip(data):
    """Unpack the data to a usable format."""
    connHost, rest = common.getNS(data)
    if isinstance(connHost, bytes):
        connHost = connHost.decode('utf-8')
    connPort = int(struct.unpack('>L', rest[:4])[0])
    origHost, rest = common.getNS(rest[4:])
    if isinstance(origHost, bytes):
        origHost = origHost.decode('utf-8')
    origPort = int(struct.unpack('>L', rest[:4])[0])
    return ((connHost, connPort), (origHost, origPort))