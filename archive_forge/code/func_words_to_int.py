import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def words_to_int(words, dialect=None):
    if dialect is None:
        dialect = DEFAULT_EUI64_DIALECT
    return _words_to_int(words, dialect.word_size, dialect.num_words)