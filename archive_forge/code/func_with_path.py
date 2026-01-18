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
def with_path(self, path, *, encoded=False):
    """Return a new URL with path replaced."""
    if not encoded:
        path = self._PATH_QUOTER(path)
        if self.is_absolute():
            path = self._normalize_path(path)
    if len(path) > 0 and path[0] != '/':
        path = '/' + path
    return URL(self._val._replace(path=path, query='', fragment=''), encoded=True)