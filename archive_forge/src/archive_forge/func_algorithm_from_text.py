from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
def algorithm_from_text(text):
    """Convert text into a DNSSEC algorithm value.

    Returns an ``int``.
    """
    value = _algorithm_by_text.get(text.upper())
    if value is None:
        value = int(text)
    return value