import dns.exception
import dns.inet
import dns.rdata
import dns.tokenizer
class AAAA(dns.rdata.Rdata):
    """AAAA record.

    @ivar address: an IPv6 address
    @type address: string (in the standard IPv6 format)"""
    __slots__ = ['address']

    def __init__(self, rdclass, rdtype, address):
        super(AAAA, self).__init__(rdclass, rdtype)
        dns.inet.inet_pton(dns.inet.AF_INET6, address)
        self.address = address

    def to_text(self, origin=None, relativize=True, **kw):
        return self.address

    @classmethod
    def from_text(cls, rdclass, rdtype, tok, origin=None, relativize=True):
        address = tok.get_identifier()
        tok.get_eol()
        return cls(rdclass, rdtype, address)

    def to_wire(self, file, compress=None, origin=None):
        file.write(dns.inet.inet_pton(dns.inet.AF_INET6, self.address))

    @classmethod
    def from_wire(cls, rdclass, rdtype, wire, current, rdlen, origin=None):
        address = dns.inet.inet_ntop(dns.inet.AF_INET6, wire[current:current + rdlen])
        return cls(rdclass, rdtype, address)