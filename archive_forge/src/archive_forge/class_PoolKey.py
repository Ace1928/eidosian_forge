from __future__ import annotations
import functools
import logging
import typing
import warnings
from types import TracebackType
from urllib.parse import urljoin
from ._collections import HTTPHeaderDict, RecentlyUsedContainer
from ._request_methods import RequestMethods
from .connection import ProxyConfig
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, port_by_scheme
from .exceptions import (
from .response import BaseHTTPResponse
from .util.connection import _TYPE_SOCKET_OPTIONS
from .util.proxy import connection_requires_http_tunnel
from .util.retry import Retry
from .util.timeout import Timeout
from .util.url import Url, parse_url
class PoolKey(typing.NamedTuple):
    """
    All known keyword arguments that could be provided to the pool manager, its
    pools, or the underlying connections.

    All custom key schemes should include the fields in this key at a minimum.
    """
    key_scheme: str
    key_host: str
    key_port: int | None
    key_timeout: Timeout | float | int | None
    key_retries: Retry | bool | int | None
    key_block: bool | None
    key_source_address: tuple[str, int] | None
    key_key_file: str | None
    key_key_password: str | None
    key_cert_file: str | None
    key_cert_reqs: str | None
    key_ca_certs: str | None
    key_ca_cert_data: str | bytes | None
    key_ssl_version: int | str | None
    key_ssl_minimum_version: ssl.TLSVersion | None
    key_ssl_maximum_version: ssl.TLSVersion | None
    key_ca_cert_dir: str | None
    key_ssl_context: ssl.SSLContext | None
    key_maxsize: int | None
    key_headers: frozenset[tuple[str, str]] | None
    key__proxy: Url | None
    key__proxy_headers: frozenset[tuple[str, str]] | None
    key__proxy_config: ProxyConfig | None
    key_socket_options: _TYPE_SOCKET_OPTIONS | None
    key__socks_options: frozenset[tuple[str, str]] | None
    key_assert_hostname: bool | str | None
    key_assert_fingerprint: str | None
    key_server_hostname: str | None
    key_blocksize: int | None