import hashlib
import dns.enum
class DigestScheme(dns.enum.IntEnum):
    """ZONEMD Scheme"""
    SIMPLE = 1

    @classmethod
    def _maximum(cls):
        return 255