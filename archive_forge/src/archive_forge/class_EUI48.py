import dns.rdtypes.euibase
class EUI48(dns.rdtypes.euibase.EUIBase):
    """EUI48 record

    @ivar fingerprint: 48-bit Extended Unique Identifier (EUI-48)
    @type fingerprint: string
    @see: rfc7043.txt"""
    byte_len = 6
    text_len = byte_len * 3 - 1