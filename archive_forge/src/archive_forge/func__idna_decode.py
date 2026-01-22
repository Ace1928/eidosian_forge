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
@functools.lru_cache(_MAXCACHE)
def _idna_decode(raw):
    try:
        return idna.decode(raw.encode('ascii'))
    except UnicodeError:
        return raw.encode('ascii').decode('idna')