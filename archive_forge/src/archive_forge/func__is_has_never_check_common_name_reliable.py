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
def _is_has_never_check_common_name_reliable(openssl_version: str, openssl_version_number: int, implementation_name: str, version_info: _TYPE_VERSION_INFO, pypy_version_info: _TYPE_VERSION_INFO | None) -> bool:
    is_openssl = openssl_version.startswith('OpenSSL ')
    is_openssl_issue_14579_fixed = openssl_version_number >= 269488335
    return is_openssl and (is_openssl_issue_14579_fixed or _is_bpo_43522_fixed(implementation_name, version_info, pypy_version_info))