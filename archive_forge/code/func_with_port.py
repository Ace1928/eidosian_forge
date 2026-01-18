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
def with_port(self, port):
    """Return a new URL with port replaced.

        Clear port to default if None is passed.

        """
    if port is not None:
        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError(f'port should be int or None, got {type(port)}')
        if port < 0 or port > 65535:
            raise ValueError(f'port must be between 0 and 65535, got {port}')
    if not self.is_absolute():
        raise ValueError('port replacement is not allowed for relative URLs')
    val = self._val
    return URL(self._val._replace(netloc=self._make_netloc(val.username, val.password, val.hostname, port)), encoded=True)