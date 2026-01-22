import struct
import base64
import binascii
import dns.exception
import dns.rdata
import dns.rdatatype
HIP record

    @ivar hit: the host identity tag
    @type hit: string
    @ivar algorithm: the public key cryptographic algorithm
    @type algorithm: int
    @ivar key: the public key
    @type key: string
    @ivar servers: the rendezvous servers
    @type servers: list of dns.name.Name objects
    @see: RFC 5205