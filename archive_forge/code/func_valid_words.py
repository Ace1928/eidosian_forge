import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def valid_words(words, dialect=None):
    if dialect is None:
        dialect = DEFAULT_EUI64_DIALECT
    return _valid_words(words, dialect.word_size, dialect.num_words)