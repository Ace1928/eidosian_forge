import dns.rdtypes.nsbase
class DNAME(dns.rdtypes.nsbase.UncompressedNS):
    """DNAME record"""

    def to_digestable(self, origin=None):
        return self.target.to_digestable(origin)