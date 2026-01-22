from __future__ import absolute_import
import math
import struct
import dns.inet
class ECSOption(Option):
    """EDNS Client Subnet (ECS, RFC7871)"""

    def __init__(self, address, srclen=None, scopelen=0):
        """*address*, a ``text``, is the client address information.

        *srclen*, an ``int``, the source prefix length, which is the
        leftmost number of bits of the address to be used for the
        lookup.  The default is 24 for IPv4 and 56 for IPv6.

        *scopelen*, an ``int``, the scope prefix length.  This value
        must be 0 in queries, and should be set in responses.
        """
        super(ECSOption, self).__init__(ECS)
        af = dns.inet.af_for_address(address)
        if af == dns.inet.AF_INET6:
            self.family = 2
            if srclen is None:
                srclen = 56
        elif af == dns.inet.AF_INET:
            self.family = 1
            if srclen is None:
                srclen = 24
        else:
            raise ValueError('Bad ip family')
        self.address = address
        self.srclen = srclen
        self.scopelen = scopelen
        addrdata = dns.inet.inet_pton(af, address)
        nbytes = int(math.ceil(srclen / 8.0))
        self.addrdata = addrdata[:nbytes]
        nbits = srclen % 8
        if nbits != 0:
            last = struct.pack('B', ord(self.addrdata[-1:]) & 255 << nbits)
            self.addrdata = self.addrdata[:-1] + last

    def to_text(self):
        return 'ECS {}/{} scope/{}'.format(self.address, self.srclen, self.scopelen)

    def to_wire(self, file):
        file.write(struct.pack('!H', self.family))
        file.write(struct.pack('!BB', self.srclen, self.scopelen))
        file.write(self.addrdata)

    @classmethod
    def from_wire(cls, otype, wire, cur, olen):
        family, src, scope = struct.unpack('!HBB', wire[cur:cur + 4])
        cur += 4
        addrlen = int(math.ceil(src / 8.0))
        if family == 1:
            af = dns.inet.AF_INET
            pad = 4 - addrlen
        elif family == 2:
            af = dns.inet.AF_INET6
            pad = 16 - addrlen
        else:
            raise ValueError('unsupported family')
        addr = dns.inet.inet_ntop(af, wire[cur:cur + addrlen] + b'\x00' * pad)
        return cls(addr, src, scope)

    def _cmp(self, other):
        if self.addrdata == other.addrdata:
            return 0
        if self.addrdata > other.addrdata:
            return 1
        return -1