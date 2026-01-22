import struct
import dns.exception
import dns.rdata
import dns.tokenizer
CAA (Certification Authority Authorization) record

    @ivar flags: the flags
    @type flags: int
    @ivar tag: the tag
    @type tag: string
    @ivar value: the value
    @type value: string
    @see: RFC 6844