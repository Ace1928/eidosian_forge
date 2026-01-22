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
def addSOA(self, name, klass, ttl, mname, rname, serial, refresh, retry, expire, minimum):
    self.addRRheader(name, Type.SOA, klass, ttl)
    self.addname(mname)
    self.addname(rname)
    self.add32bit(serial)
    self.add32bit(refresh)
    self.add32bit(retry)
    self.add32bit(expire)
    self.add32bit(minimum)
    self.endRR()