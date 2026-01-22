from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class BadLabelType(dns.exception.FormError):
    """The label type in DNS name wire format is unknown."""