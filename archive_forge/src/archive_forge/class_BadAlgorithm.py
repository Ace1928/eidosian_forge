import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
class BadAlgorithm(dns.exception.DNSException):
    """The TSIG algorithm does not match the key."""