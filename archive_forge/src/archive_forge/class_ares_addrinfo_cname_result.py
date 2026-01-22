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
class ares_addrinfo_cname_result(AresResult):
    __slots__ = ('ttl', 'alias', 'name')

    def __init__(self, ares_cname):
        self.ttl = ares_cname.ttl
        self.alias = maybe_str(_ffi.string(ares_cname.alias))
        self.name = maybe_str(_ffi.string(ares_cname.name))