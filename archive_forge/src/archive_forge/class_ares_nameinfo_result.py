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
class ares_nameinfo_result(AresResult):
    __slots__ = ('node', 'service')

    def __init__(self, node, service):
        self.node = maybe_str(_ffi.string(node))
        self.service = maybe_str(_ffi.string(service)) if service != _ffi.NULL else None