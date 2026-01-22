import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
class mac_eui48(object):
    """A standard IEEE EUI-48 dialect class."""
    word_size = 8
    num_words = width // word_size
    max_word = 2 ** word_size - 1
    word_sep = '-'
    word_fmt = '%.2X'
    word_base = 16