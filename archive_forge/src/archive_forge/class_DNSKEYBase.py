import base64
import struct
import dns.exception
import dns.dnssec
import dns.rdata
class DNSKEYBase(dns.rdata.Rdata):
    """Base class for rdata that is like a DNSKEY record

    @ivar flags: the key flags
    @type flags: int
    @ivar protocol: the protocol for which this key may be used
    @type protocol: int
    @ivar algorithm: the algorithm used for the key
    @type algorithm: int
    @ivar key: the public key
    @type key: string"""
    __slots__ = ['flags', 'protocol', 'algorithm', 'key']

    def __init__(self, rdclass, rdtype, flags, protocol, algorithm, key):
        super(DNSKEYBase, self).__init__(rdclass, rdtype)
        self.flags = flags
        self.protocol = protocol
        self.algorithm = algorithm
        self.key = key

    def to_text(self, origin=None, relativize=True, **kw):
        return '%d %d %d %s' % (self.flags, self.protocol, self.algorithm, dns.rdata._base64ify(self.key))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        flags = tok.get_uint16()
        protocol = tok.get_uint8()
        algorithm = dns.dnssec.algorithm_from_text(tok.get_string())
        chunks = []
        while 1:
            t = tok.get().unescape()
            if t.is_eol_or_eof():
                break
            if not t.is_identifier():
                raise dns.exception.SyntaxError
            chunks.append(t.value.encode())
        b64 = b''.join(chunks)
        key = base64.b64decode(b64)
        return cls(rdclass, rdtype, flags, protocol, algorithm, key)

    def to_wire(self, file, compress=None, origin=None):
        header = struct.pack('!HBB', self.flags, self.protocol, self.algorithm)
        file.write(header)
        file.write(self.key)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        if rdlen < 4:
            raise dns.exception.FormError
        header = struct.unpack('!HBB', wire[current:current + 4])
        current += 4
        rdlen -= 4
        key = wire[current:current + rdlen].unwrap()
        return cls(rdclass, rdtype, header[0], header[1], header[2], key)

    def flags_to_text_set(self):
        """Convert a DNSKEY flags value to set texts
        @rtype: set([string])"""
        return flags_to_text_set(self.flags)