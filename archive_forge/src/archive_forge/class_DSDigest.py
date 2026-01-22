import dns.enum
class DSDigest(dns.enum.IntEnum):
    """DNSSEC Delegation Signer Digest Algorithm"""
    NULL = 0
    SHA1 = 1
    SHA256 = 2
    GOST = 3
    SHA384 = 4

    @classmethod
    def _maximum(cls):
        return 255