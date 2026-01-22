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
@_ffi.def_extern()
def _query_cb(arg, status, timeouts, abuf, alen):
    callback, query_type = _ffi.from_handle(arg)
    _global_set.discard(arg)
    if status == _lib.ARES_SUCCESS:
        if query_type == _lib.T_ANY:
            result = []
            for qtype in (_lib.T_A, _lib.T_AAAA, _lib.T_CAA, _lib.T_CNAME, _lib.T_MX, _lib.T_NAPTR, _lib.T_NS, _lib.T_PTR, _lib.T_SOA, _lib.T_SRV, _lib.T_TXT):
                r, status = parse_result(qtype, abuf, alen)
                if status not in (None, _lib.ARES_ENODATA, _lib.ARES_EBADRESP):
                    result = None
                    break
                if r is not None:
                    if isinstance(r, collections.abc.Iterable):
                        result.extend(r)
                    else:
                        result.append(r)
            else:
                status = None
        else:
            result, status = parse_result(query_type, abuf, alen)
    else:
        result = None
    callback(result, status)