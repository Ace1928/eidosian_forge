import struct
from twisted.internet import defer
from twisted.protocols import basic
from twisted.python import failure, log
def unpackAddress(self, packed):
    addr, port = packed.split(':')
    addr = self.dottedQuadFromHexString(addr)
    port = int(port, 16)
    return (addr, port)