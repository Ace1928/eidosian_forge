import hashlib
import hmac
import struct
import dns.exception
import dns.rdataclass
import dns.name
from ._compat import long, string_types, text_type
class PeerBadKey(PeerError):
    """The peer didn't know the key we used"""