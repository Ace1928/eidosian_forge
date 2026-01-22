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
class ares_query_cname_result(AresResult):
    __slots__ = ('cname', 'ttl')
    type = 'CNAME'

    def __init__(self, host):
        self.cname = maybe_str(_ffi.string(host.h_name))
        self.ttl = -1