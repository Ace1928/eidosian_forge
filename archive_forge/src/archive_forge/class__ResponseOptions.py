from __future__ import annotations
import typing
from .util.connection import _TYPE_SOCKET_OPTIONS
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from .util.url import Url
class _ResponseOptions(typing.NamedTuple):
    request_method: str
    request_url: str
    preload_content: bool
    decode_content: bool
    enforce_content_length: bool