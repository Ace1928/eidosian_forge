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
class RRpacker(Packer):

    def __init__(self):
        Packer.__init__(self)
        self.rdstart = None

    def addRRheader(self, name, RRtype, klass, ttl, *rest):
        self.addname(name)
        self.add16bit(RRtype)
        self.add16bit(klass)
        self.add32bit(ttl)
        if rest:
            if rest[1:]:
                raise TypeError('too many args')
            rdlength = rest[0]
        else:
            rdlength = 0
        self.add16bit(rdlength)
        self.rdstart = len(self.buf)

    def patchrdlength(self):
        rdlength = unpack16bit(self.buf[self.rdstart - 2:self.rdstart])
        if rdlength == len(self.buf) - self.rdstart:
            return
        rdata = self.buf[self.rdstart:]
        save_buf = self.buf
        ok = 0
        try:
            self.buf = self.buf[:self.rdstart - 2]
            self.add16bit(len(rdata))
            self.buf = self.buf + rdata
            ok = 1
        finally:
            if not ok:
                self.buf = save_buf

    def endRR(self):
        if self.rdstart is not None:
            self.patchrdlength()
        self.rdstart = None

    def getbuf(self):
        if self.rdstart is not None:
            self.patchrdlength()
        return Packer.getbuf(self)

    def addCNAME(self, name, klass, ttl, cname):
        self.addRRheader(name, Type.CNAME, klass, ttl)
        self.addname(cname)
        self.endRR()

    def addHINFO(self, name, klass, ttl, cpu, os):
        self.addRRheader(name, Type.HINFO, klass, ttl)
        self.addstring(cpu)
        self.addstring(os)
        self.endRR()

    def addMX(self, name, klass, ttl, preference, exchange):
        self.addRRheader(name, Type.MX, klass, ttl)
        self.add16bit(preference)
        self.addname(exchange)
        self.endRR()

    def addNS(self, name, klass, ttl, nsdname):
        self.addRRheader(name, Type.NS, klass, ttl)
        self.addname(nsdname)
        self.endRR()

    def addPTR(self, name, klass, ttl, ptrdname):
        self.addRRheader(name, Type.PTR, klass, ttl)
        self.addname(ptrdname)
        self.endRR()

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

    def addTXT(self, name, klass, ttl, tlist):
        self.addRRheader(name, Type.TXT, klass, ttl)
        if type(tlist) is bytes or type(tlist) is str:
            tlist = [tlist]
        for txtdata in tlist:
            self.addstring(txtdata)
        self.endRR()

    def addSPF(self, name, klass, ttl, tlist):
        self.addRRheader(name, Type.TXT, klass, ttl)
        if type(tlist) is bytes or type(tlist) is str:
            tlist = [tlist]
        for txtdata in tlist:
            self.addstring(txtdata)
        self.endRR()

    def addA(self, name, klass, ttl, address):
        self.addRRheader(name, Type.A, klass, ttl)
        self.addaddr(address)
        self.endRR()

    def addWKS(self, name, ttl, address, protocol, bitmap):
        self.addRRheader(name, Type.WKS, Class.IN, ttl)
        self.addaddr(address)
        self.addbyte(chr(protocol))
        self.addbytes(bitmap)
        self.endRR()

    def addSRV(self):
        raise NotImplementedError