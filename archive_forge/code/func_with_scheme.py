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
def with_scheme(self, scheme):
    """Return a new URL with scheme replaced."""
    if not isinstance(scheme, str):
        raise TypeError('Invalid scheme type')
    if not self.is_absolute():
        raise ValueError('scheme replacement is not allowed for relative URLs')
    return URL(self._val._replace(scheme=scheme.lower()), encoded=True)