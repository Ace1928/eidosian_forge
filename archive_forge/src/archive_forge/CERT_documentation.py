import struct
import base64
import dns.exception
import dns.dnssec
import dns.rdata
import dns.tokenizer
CERT record

    @ivar certificate_type: certificate type
    @type certificate_type: int
    @ivar key_tag: key tag
    @type key_tag: int
    @ivar algorithm: algorithm
    @type algorithm: int
    @ivar certificate: the certificate or CRL
    @type certificate: string
    @see: RFC 2538