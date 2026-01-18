import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def valid_bin(bin_val, dialect=None):
    if dialect is None:
        dialect = DEFAULT_EUI64_DIALECT
    return _valid_bin(bin_val, width)