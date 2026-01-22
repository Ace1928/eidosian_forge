import struct
import binascii
import dns.rdata
import dns.rdatatype
class SSHFP(dns.rdata.Rdata):
    """SSHFP record

    @ivar algorithm: the algorithm
    @type algorithm: int
    @ivar fp_type: the digest type
    @type fp_type: int
    @ivar fingerprint: the fingerprint
    @type fingerprint: string
    @see: draft-ietf-secsh-dns-05.txt"""
    __slots__ = ['algorithm', 'fp_type', 'fingerprint']

    def __init__(self, rdclass, rdtype, algorithm, fp_type, fingerprint):
        super(SSHFP, self).__init__(rdclass, rdtype)
        self.algorithm = algorithm
        self.fp_type = fp_type
        self.fingerprint = fingerprint

    def to_text(self, origin=None, relativize=True, **kw):
        return '%d %d %s' % (self.algorithm, self.fp_type, dns.rdata._hexify(self.fingerprint, chunksize=128))

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        algorithm = tok.get_uint8()
        fp_type = tok.get_uint8()
        chunks = []
        while 1:
            t = tok.get().unescape()
            if t.is_eol_or_eof():
                break
            if not t.is_identifier():
                raise dns.exception.SyntaxError
            chunks.append(t.value.encode())
        fingerprint = b''.join(chunks)
        fingerprint = binascii.unhexlify(fingerprint)
        return cls(rdclass, rdtype, algorithm, fp_type, fingerprint)

    def to_wire(self, file, compress=None, origin=None):
        header = struct.pack('!BB', self.algorithm, self.fp_type)
        file.write(header)
        file.write(self.fingerprint)

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        header = struct.unpack('!BB', wire[current:current + 2])
        current += 2
        rdlen -= 2
        fingerprint = wire[current:current + rdlen].unwrap()
        return cls(rdclass, rdtype, header[0], header[1], fingerprint)