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
def addTXT(self, name, klass, ttl, tlist):
    self.addRRheader(name, Type.TXT, klass, ttl)
    if type(tlist) is bytes or type(tlist) is str:
        tlist = [tlist]
    for txtdata in tlist:
        self.addstring(txtdata)
    self.endRR()