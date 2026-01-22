import types
import socket
from . import Type
from . import Class
from . import Opcode
from . import Status
import DNS
from .Base import DNSError
from struct import pack as struct_pack
from struct import unpack as struct_unpack
from socket import inet_ntoa, inet_aton, inet_ntop, AF_INET6
class Hpacker(Packer):

    def addHeader(self, id, qr, opcode, aa, tc, rd, ra, z, rcode, qdcount, ancount, nscount, arcount):
        self.add16bit(id)
        self.add16bit((qr & 1) << 15 | (opcode & 15) << 11 | (aa & 1) << 10 | (tc & 1) << 9 | (rd & 1) << 8 | (ra & 1) << 7 | (z & 7) << 4 | rcode & 15)
        self.add16bit(qdcount)
        self.add16bit(ancount)
        self.add16bit(nscount)
        self.add16bit(arcount)