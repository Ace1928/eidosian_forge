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
def with_user(self, user):
    """Return a new URL with user replaced.

        Autoencode user if needed.

        Clear user/password if user is None.

        """
    val = self._val
    if user is None:
        password = None
    elif isinstance(user, str):
        user = self._QUOTER(user)
        password = val.password
    else:
        raise TypeError('Invalid user type')
    if not self.is_absolute():
        raise ValueError('user replacement is not allowed for relative URLs')
    return URL(self._val._replace(netloc=self._make_netloc(user, password, val.hostname, val.port)), encoded=True)