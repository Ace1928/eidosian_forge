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
@staticmethod
def _query_var(v):
    cls = type(v)
    if issubclass(cls, str):
        return v
    if issubclass(cls, float):
        if math.isinf(v):
            raise ValueError("float('inf') is not supported")
        if math.isnan(v):
            raise ValueError("float('nan') is not supported")
        return str(float(v))
    if issubclass(cls, int) and cls is not bool:
        return str(int(v))
    raise TypeError('Invalid variable type: value should be str, int or float, got {!r} of type {}'.format(v, cls))