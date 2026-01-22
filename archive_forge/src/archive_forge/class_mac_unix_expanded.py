import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class mac_unix_expanded(mac_unix):
    """A UNIX-style MAC address dialect class with leading zeroes."""
    word_fmt = '%.2x'