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
def _get_str_query(self, *args, **kwargs):
    if kwargs:
        if len(args) > 0:
            raise ValueError('Either kwargs or single query parameter must be present')
        query = kwargs
    elif len(args) == 1:
        query = args[0]
    else:
        raise ValueError('Either kwargs or single query parameter must be present')
    if query is None:
        query = None
    elif isinstance(query, Mapping):
        quoter = self._QUERY_PART_QUOTER
        query = '&'.join(self._query_seq_pairs(quoter, query.items()))
    elif isinstance(query, str):
        query = self._QUERY_QUOTER(query)
    elif isinstance(query, (bytes, bytearray, memoryview)):
        raise TypeError('Invalid query type: bytes, bytearray and memoryview are forbidden')
    elif isinstance(query, Sequence):
        quoter = self._QUERY_PART_QUOTER
        query = '&'.join((quoter(k) + '=' + quoter(self._query_var(v)) for k, v in query))
    else:
        raise TypeError('Invalid query type: only str, mapping or sequence of (key, value) pairs is allowed')
    return query