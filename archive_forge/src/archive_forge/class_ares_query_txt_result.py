from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
class ares_query_txt_result(AresResult):
    __slots__ = ('text', 'ttl')
    type = 'TXT'

    def __init__(self, txt_chunk):
        self.text = maybe_str(txt_chunk.text)
        self.ttl = -1