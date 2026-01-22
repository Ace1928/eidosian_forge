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
def addname(self, name):
    nlist = []
    for label in name.split('.'):
        if not label:
            pass
        else:
            nlist.append(label)
    keys = []
    for i in range(len(nlist)):
        key = '.'.join(nlist[i:])
        key = key.upper()
        keys.append(key)
        if key in self.index:
            pointer = self.index[key]
            break
    else:
        i = len(nlist)
        pointer = None
    offset = len(self.buf)
    index = []
    if DNS.LABEL_UTF8:
        enc = 'utf8'
    else:
        enc = DNS.LABEL_ENCODING
    buf = bytes('', enc)
    for j in range(i):
        label = nlist[j]
        try:
            label = label.encode(enc)
        except UnicodeEncodeError:
            if not DNS.LABEL_UTF8:
                raise
            if not label.startswith('\ufeff'):
                label = '\ufeff' + label
            label = label.encode(enc)
        n = len(label)
        if n > 63:
            raise PackError('label too long')
        if offset + len(buf) < 16383:
            index.append((keys[j], offset + len(buf)))
        else:
            print('DNS.Lib.Packer.addname:')
            print('warning: pointer too big')
        buf = buf + bytes([n]) + label
    if pointer:
        buf = buf + pack16bit(pointer | 49152)
    else:
        buf = buf + bytes('\x00', enc)
    self.buf = self.buf + buf
    for key, value in index:
        self.index[key] = value