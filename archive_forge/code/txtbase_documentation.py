import struct
import dns.exception
import dns.rdata
import dns.tokenizer
from dns._compat import binary_type, string_types
Base class for rdata that is like a TXT record

    @ivar strings: the strings
    @type strings: list of binary
    @see: RFC 1035