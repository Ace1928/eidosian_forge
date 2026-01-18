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
def with_fragment(self, fragment):
    """Return a new URL with fragment replaced.

        Autoencode fragment if needed.

        Clear fragment to default if None is passed.

        """
    if fragment is None:
        raw_fragment = ''
    elif not isinstance(fragment, str):
        raise TypeError('Invalid fragment type')
    else:
        raw_fragment = self._FRAGMENT_QUOTER(fragment)
    if self.raw_fragment == raw_fragment:
        return self
    return URL(self._val._replace(fragment=raw_fragment), encoded=True)