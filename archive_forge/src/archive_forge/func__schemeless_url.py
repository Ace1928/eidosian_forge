from __future__ import annotations
import re
from ipaddress import AddressValueError, IPv6Address
from urllib.parse import scheme_chars
def _schemeless_url(url: str) -> str:
    double_slashes_start = url.find('//')
    if double_slashes_start == 0:
        return url[2:]
    if double_slashes_start < 2 or not url[double_slashes_start - 1] == ':' or set(url[:double_slashes_start - 1]) - scheme_chars_set:
        return url
    return url[double_slashes_start + 2:]