import functools
import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from ipaddress import ip_address
from urllib.parse import SplitResult, parse_qsl, quote, urljoin, urlsplit, urlunsplit
import idna
from multidict import MultiDict, MultiDictProxy
from ._quoting import _Quoter, _Unquoter
def _human_quote(s, unsafe):
    if not s:
        return s
    for c in '%' + unsafe:
        if c in s:
            s = s.replace(c, f'%{ord(c):02X}')
    if s.isprintable():
        return s
    return ''.join((c if c.isprintable() else quote(c) for c in s))