from __future__ import annotations
import typing
from urllib.parse import parse_qs, unquote
import idna
from ._types import QueryParamTypes, RawURL, URLTypes
from ._urlparse import urlencode, urlparse
from ._utils import primitive_value_to_str
@property
def userinfo(self) -> bytes:
    """
        The URL userinfo as a raw bytestring.
        For example: b"jo%40email.com:a%20secret".
        """
    return self._uri_reference.userinfo.encode('ascii')