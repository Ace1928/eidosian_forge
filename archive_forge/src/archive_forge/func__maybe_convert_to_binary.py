from io import BytesIO
import struct
import sys
import copy
import encodings.idna
import dns.exception
import dns.wiredata
from ._compat import long, binary_type, text_type, unichr, maybe_decode
def _maybe_convert_to_binary(label):
    """If label is ``text``, convert it to ``binary``.  If it is already
    ``binary`` just return it.

    """
    if isinstance(label, binary_type):
        return label
    if isinstance(label, text_type):
        return label.encode()
    raise ValueError