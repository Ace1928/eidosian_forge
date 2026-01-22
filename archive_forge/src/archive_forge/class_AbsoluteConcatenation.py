from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
class AbsoluteConcatenation(dns.exception.DNSException):
    """An attempt was made to append anything other than the
    empty name to an absolute DNS name."""