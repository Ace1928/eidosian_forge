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
def addbytes(self, abytes):
    if DNS.LABEL_UTF8:
        enc = 'utf8'
    else:
        enc = DNS.LABEL_ENCODING
    self.buf = self.buf + bytes(abytes, enc)