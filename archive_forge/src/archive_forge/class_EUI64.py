import dns.rdtypes.euibase
class EUI64(dns.rdtypes.euibase.EUIBase):
    """EUI64 record

    @ivar fingerprint: 64-bit Extended Unique Identifier (EUI-64)
    @type fingerprint: string
    @see: rfc7043.txt"""
    byte_len = 8
    text_len = byte_len * 3 - 1