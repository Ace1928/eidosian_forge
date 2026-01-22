from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class BadPointer(dns.exception.FormError):
    """A DNS compression pointer points forward instead of backward."""