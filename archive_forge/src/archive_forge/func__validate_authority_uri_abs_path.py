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
def _validate_authority_uri_abs_path(host, path):
    """Ensure that path in URL with authority starts with a leading slash.

        Raise ValueError if not.
        """
    if len(host) > 0 and len(path) > 0 and (not path.startswith('/')):
        raise ValueError("Path in a URL with authority should start with a slash ('/') if set")