import struct as _struct
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class ipv6_compact(object):
    """An IPv6 dialect class - compact form."""
    word_fmt = '%x'
    compact = True