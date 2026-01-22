import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _strip_ipv6_iface(enc_name: bytes) -> bytes:
    """Remove interface scope from IPv6 address."""
    enc_name, percent, _ = enc_name.partition(b'%')
    if percent:
        assert enc_name.startswith(b'['), enc_name
        enc_name += b']'
    return enc_name