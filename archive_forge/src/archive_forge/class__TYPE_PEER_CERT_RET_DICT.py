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
class _TYPE_PEER_CERT_RET_DICT(TypedDict, total=False):
    subjectAltName: tuple[tuple[str, str], ...]
    subject: tuple[tuple[tuple[str, str], ...], ...]
    serialNumber: str