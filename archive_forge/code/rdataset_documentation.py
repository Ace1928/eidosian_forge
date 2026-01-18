import random
from io import StringIO
import struct
import dns.exception
import dns.rdatatype
import dns.rdataclass
import dns.rdata
import dns.set
from ._compat import string_types
Returns ``True`` if this rdataset matches the specified class,
        type, and covers.
        