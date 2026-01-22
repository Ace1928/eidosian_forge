import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
class ProcServerMixin:
    """Implements lookup() to grab entries for responses from /proc/net/tcp"""
    SYSTEM_NAME = 'LINUX'
    try:
        from pwd import getpwuid

        def getUsername(self, uid, getpwuid=getpwuid):
            return getpwuid(uid)[0]
        del getpwuid
    except ImportError:

        def getUsername(self, uid, getpwuid=None):
            raise IdentError()

    def entries(self):
        with open('/proc/net/tcp') as f:
            f.readline()
            for L in f:
                yield L.strip()

    def dottedQuadFromHexString(self, hexstr):
        return '.'.join(map(str, struct.unpack('4B', struct.pack('=L', int(hexstr, 16)))))

    def unpackAddress(self, packed):
        addr, port = packed.split(':')
        addr = self.dottedQuadFromHexString(addr)
        port = int(port, 16)
        return (addr, port)

    def parseLine(self, line):
        parts = line.strip().split()
        localAddr, localPort = self.unpackAddress(parts[1])
        remoteAddr, remotePort = self.unpackAddress(parts[2])
        uid = int(parts[7])
        return ((localAddr, localPort), (remoteAddr, remotePort), uid)

    def lookup(self, serverAddress, clientAddress):
        for ent in self.entries():
            localAddr, remoteAddr, uid = self.parseLine(ent)
            if remoteAddr == clientAddress and localAddr[1] == serverAddress[1]:
                return (self.SYSTEM_NAME, self.getUsername(uid))
        raise NoUser()