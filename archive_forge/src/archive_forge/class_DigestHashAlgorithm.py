import hashlib
import dns.enum
class DigestHashAlgorithm(dns.enum.IntEnum):
    """ZONEMD Hash Algorithm"""
    SHA384 = 1
    SHA512 = 2

    @classmethod
    def _maximum(cls):
        return 255