from __future__ import annotations
import hmac
import os
import socket
import sys
import typing
import warnings
from binascii import unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import ProxySchemeUnsupported, SSLError
from .url import _BRACELESS_IPV6_ADDRZ_RE, _IPV4_RE
def _is_bpo_43522_fixed(implementation_name: str, version_info: _TYPE_VERSION_INFO, pypy_version_info: _TYPE_VERSION_INFO | None) -> bool:
    """Return True for CPython 3.8.9+, 3.9.3+ or 3.10+ and PyPy 7.3.8+ where
    setting SSLContext.hostname_checks_common_name to False works.

    Outside of CPython and PyPy we don't know which implementations work
    or not so we conservatively use our hostname matching as we know that works
    on all implementations.

    https://github.com/urllib3/urllib3/issues/2192#issuecomment-821832963
    https://foss.heptapod.net/pypy/pypy/-/issues/3539
    """
    if implementation_name == 'pypy':
        return pypy_version_info >= (7, 3, 8)
    elif implementation_name == 'cpython':
        major_minor = version_info[:2]
        micro = version_info[2]
        return major_minor == (3, 8) and micro >= 9 or (major_minor == (3, 9) and micro >= 3) or major_minor >= (3, 10)
    else:
        return False