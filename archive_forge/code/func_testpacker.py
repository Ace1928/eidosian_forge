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
def testpacker():
    N = 2500
    R = list(range(N))
    import timing
    timing.start()
    for i in R:
        p = Packer()
        p.addaddr('192.168.0.1')
        p.addbytes('*' * 20)
        p.addname('f.ISI.ARPA')
        p.addbytes('*' * 8)
        p.addname('Foo.F.isi.arpa')
        p.addbytes('*' * 18)
        p.addname('arpa')
        p.addbytes('*' * 26)
        p.addname('')
    timing.finish()
    print(timing.milli(), 'ms total for packing')
    print(round(timing.milli() / i, 4), 'ms per packing')
    u = Unpacker(p.buf)
    u.getaddr()
    u.getbytes(20)
    u.getname()
    u.getbytes(8)
    u.getname()
    u.getbytes(18)
    u.getname()
    u.getbytes(26)
    u.getname()
    timing.start()
    for i in R:
        u = Unpacker(p.buf)
        res = (u.getaddr(), u.getbytes(20), u.getname(), u.getbytes(8), u.getname(), u.getbytes(18), u.getname(), u.getbytes(26), u.getname())
    timing.finish()
    print(timing.milli(), 'ms total for unpacking')
    print(round(timing.milli() / i, 4), 'ms per unpacking')