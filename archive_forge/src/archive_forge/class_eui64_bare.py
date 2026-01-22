import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class eui64_bare(eui64_base):
    """A bare (no delimiters) MAC address dialect class."""
    word_size = 64
    num_words = width // word_size
    word_sep = ''
    word_fmt = '%.16X'
    word_base = 16