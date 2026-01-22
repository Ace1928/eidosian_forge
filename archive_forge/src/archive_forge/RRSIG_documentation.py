import base64
import calendar
import struct
import time
import dns.dnssec
import dns.exception
import dns.rdata
import dns.rdatatype
RRSIG record

    @ivar type_covered: the rdata type this signature covers
    @type type_covered: int
    @ivar algorithm: the algorithm used for the sig
    @type algorithm: int
    @ivar labels: number of labels
    @type labels: int
    @ivar original_ttl: the original TTL
    @type original_ttl: long
    @ivar expiration: signature expiration time
    @type expiration: long
    @ivar inception: signature inception time
    @type inception: long
    @ivar key_tag: the key tag
    @type key_tag: int
    @ivar signer: the signer
    @type signer: dns.name.Name object
    @ivar signature: the signature
    @type signature: string