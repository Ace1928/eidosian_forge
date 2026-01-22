from __future__ import annotations
import typing
from .util.connection import _TYPE_SOCKET_OPTIONS
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from .util.url import Url
class BaseHTTPConnection(Protocol):
    default_port: typing.ClassVar[int]
    default_socket_options: typing.ClassVar[_TYPE_SOCKET_OPTIONS]
    host: str
    port: int
    timeout: None | float
    blocksize: int
    source_address: tuple[str, int] | None
    socket_options: _TYPE_SOCKET_OPTIONS | None
    proxy: Url | None
    proxy_config: ProxyConfig | None
    is_verified: bool
    proxy_is_verified: bool | None

    def __init__(self, host: str, port: int | None=None, *, timeout: _TYPE_TIMEOUT=_DEFAULT_TIMEOUT, source_address: tuple[str, int] | None=None, blocksize: int=8192, socket_options: _TYPE_SOCKET_OPTIONS | None=..., proxy: Url | None=None, proxy_config: ProxyConfig | None=None) -> None:
        ...

    def set_tunnel(self, host: str, port: int | None=None, headers: typing.Mapping[str, str] | None=None, scheme: str='http') -> None:
        ...

    def connect(self) -> None:
        ...

    def request(self, method: str, url: str, body: _TYPE_BODY | None=None, headers: typing.Mapping[str, str] | None=None, *, chunked: bool=False, preload_content: bool=True, decode_content: bool=True, enforce_content_length: bool=True) -> None:
        ...

    def getresponse(self) -> BaseHTTPResponse:
        ...

    def close(self) -> None:
        ...

    @property
    def is_closed(self) -> bool:
        """Whether the connection either is brand new or has been previously closed.
            If this property is True then both ``is_connected`` and ``has_connected_to_proxy``
            properties must be False.
            """

    @property
    def is_connected(self) -> bool:
        """Whether the connection is actively connected to any origin (proxy or target)"""

    @property
    def has_connected_to_proxy(self) -> bool:
        """Whether the connection has successfully connected to its proxy.
            This returns False if no proxy is in use. Used to determine whether
            errors are coming from the proxy layer or from tunnelling to the target origin.
            """