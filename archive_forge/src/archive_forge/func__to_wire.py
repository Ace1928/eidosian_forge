import struct
import dns.immutable
import dns.rdata
def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
    file.write(struct.pack('!H', self.preference))
    file.write(dns.ipv4.inet_aton(self.locator32))