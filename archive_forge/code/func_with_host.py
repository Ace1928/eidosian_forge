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
def with_host(self, host):
    """Return a new URL with host replaced.

        Autoencode host if needed.

        Changing host for relative URLs is not allowed, use .join()
        instead.

        """
    if not isinstance(host, str):
        raise TypeError('Invalid host type')
    if not self.is_absolute():
        raise ValueError('host replacement is not allowed for relative URLs')
    if not host:
        raise ValueError('host removing is not allowed')
    val = self._val
    return URL(self._val._replace(netloc=self._make_netloc(val.username, val.password, host, val.port)), encoded=True)