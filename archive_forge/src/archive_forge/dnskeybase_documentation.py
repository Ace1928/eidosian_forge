import base64
import struct
import dns.exception
import dns.dnssec
import dns.rdata
Convert a DNSKEY flags value to set texts
        @rtype: set([string])