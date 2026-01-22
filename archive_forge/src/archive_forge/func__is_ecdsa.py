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
def _is_ecdsa(algorithm):
    return _have_ecdsa and algorithm in (ECDSAP256SHA256, ECDSAP384SHA384)