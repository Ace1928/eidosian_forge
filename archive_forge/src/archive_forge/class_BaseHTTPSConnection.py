from __future__ import annotations
import typing
from .util.connection import _TYPE_SOCKET_OPTIONS
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT
from .util.url import Url
class BaseHTTPSConnection(BaseHTTPConnection, Protocol):
    default_port: typing.ClassVar[int]
    default_socket_options: typing.ClassVar[_TYPE_SOCKET_OPTIONS]
    cert_reqs: int | str | None
    assert_hostname: None | str | Literal[False]
    assert_fingerprint: str | None
    ssl_context: ssl.SSLContext | None
    ca_certs: str | None
    ca_cert_dir: str | None
    ca_cert_data: None | str | bytes
    ssl_minimum_version: int | None
    ssl_maximum_version: int | None
    ssl_version: int | str | None
    cert_file: str | None
    key_file: str | None
    key_password: str | None

    def __init__(self, host: str, port: int | None=None, *, timeout: _TYPE_TIMEOUT=_DEFAULT_TIMEOUT, source_address: tuple[str, int] | None=None, blocksize: int=16384, socket_options: _TYPE_SOCKET_OPTIONS | None=..., proxy: Url | None=None, proxy_config: ProxyConfig | None=None, cert_reqs: int | str | None=None, assert_hostname: None | str | Literal[False]=None, assert_fingerprint: str | None=None, server_hostname: str | None=None, ssl_context: ssl.SSLContext | None=None, ca_certs: str | None=None, ca_cert_dir: str | None=None, ca_cert_data: None | str | bytes=None, ssl_minimum_version: int | None=None, ssl_maximum_version: int | None=None, ssl_version: int | str | None=None, cert_file: str | None=None, key_file: str | None=None, key_password: str | None=None) -> None:
        ...