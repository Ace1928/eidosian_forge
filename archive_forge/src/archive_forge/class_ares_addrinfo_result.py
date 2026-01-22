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
class ares_addrinfo_result(AresResult):
    __slots__ = ('cnames', 'nodes')

    def __init__(self, ares_addrinfo):
        self.cnames = []
        self.nodes = []
        cname_ptr = ares_addrinfo.cnames
        while cname_ptr != _ffi.NULL:
            self.cnames.append(ares_addrinfo_cname_result(cname_ptr))
            cname_ptr = cname_ptr.next
        node_ptr = ares_addrinfo.nodes
        while node_ptr != _ffi.NULL:
            self.nodes.append(ares_addrinfo_node_result(node_ptr))
            node_ptr = node_ptr.ai_next
        _lib.ares_freeaddrinfo(ares_addrinfo)