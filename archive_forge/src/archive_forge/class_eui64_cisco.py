import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class eui64_cisco(eui64_base):
    """A Cisco 'triple hextet' MAC address dialect class."""
    word_size = 16
    num_words = width // word_size
    word_sep = '.'
    word_fmt = '%.4x'
    word_base = 16