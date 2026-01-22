from __future__ import annotations
import datetime
import logging
import os
import re
import socket
import sys
import typing
import warnings
from http.client import HTTPConnection as _HTTPConnection
from http.client import HTTPException as HTTPException  # noqa: F401
from http.client import ResponseNotReady
from socket import timeout as SocketTimeout
from ._collections import HTTPHeaderDict
from .util.response import assert_header_parsing
from .util.timeout import _DEFAULT_TIMEOUT, _TYPE_TIMEOUT, Timeout
from .util.util import to_str
from .util.wait import wait_for_read
from ._base_connection import _TYPE_BODY
from ._base_connection import ProxyConfig as ProxyConfig
from ._base_connection import _ResponseOptions as _ResponseOptions
from ._version import __version__
from .exceptions import (
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection, ssl_
from .util.request import body_to_chunks
from .util.ssl_ import assert_fingerprint as _assert_fingerprint
from .util.ssl_ import (
from .util.ssl_match_hostname import CertificateError, match_hostname
from .util.url import Url
def _ssl_wrap_socket_and_match_hostname(sock: socket.socket, *, cert_reqs: None | str | int, ssl_version: None | str | int, ssl_minimum_version: int | None, ssl_maximum_version: int | None, cert_file: str | None, key_file: str | None, key_password: str | None, ca_certs: str | None, ca_cert_dir: str | None, ca_cert_data: None | str | bytes, assert_hostname: None | str | Literal[False], assert_fingerprint: str | None, server_hostname: str | None, ssl_context: ssl.SSLContext | None, tls_in_tls: bool=False) -> _WrappedAndVerifiedSocket:
    """Logic for constructing an SSLContext from all TLS parameters, passing
    that down into ssl_wrap_socket, and then doing certificate verification
    either via hostname or fingerprint. This function exists to guarantee
    that both proxies and targets have the same behavior when connecting via TLS.
    """
    default_ssl_context = False
    if ssl_context is None:
        default_ssl_context = True
        context = create_urllib3_context(ssl_version=resolve_ssl_version(ssl_version), ssl_minimum_version=ssl_minimum_version, ssl_maximum_version=ssl_maximum_version, cert_reqs=resolve_cert_reqs(cert_reqs))
    else:
        context = ssl_context
    context.verify_mode = resolve_cert_reqs(cert_reqs)
    if assert_fingerprint or assert_hostname or assert_hostname is False or ssl_.IS_PYOPENSSL or (not ssl_.HAS_NEVER_CHECK_COMMON_NAME):
        context.check_hostname = False
    if not ca_certs and (not ca_cert_dir) and (not ca_cert_data) and default_ssl_context and hasattr(context, 'load_default_certs'):
        context.load_default_certs()
    if server_hostname is not None:
        normalized = server_hostname.strip('[]')
        if '%' in normalized:
            normalized = normalized[:normalized.rfind('%')]
        if is_ipaddress(normalized):
            server_hostname = normalized
    ssl_sock = ssl_wrap_socket(sock=sock, keyfile=key_file, certfile=cert_file, key_password=key_password, ca_certs=ca_certs, ca_cert_dir=ca_cert_dir, ca_cert_data=ca_cert_data, server_hostname=server_hostname, ssl_context=context, tls_in_tls=tls_in_tls)
    try:
        if assert_fingerprint:
            _assert_fingerprint(ssl_sock.getpeercert(binary_form=True), assert_fingerprint)
        elif context.verify_mode != ssl.CERT_NONE and (not context.check_hostname) and (assert_hostname is not False):
            cert: _TYPE_PEER_CERT_RET_DICT = ssl_sock.getpeercert()
            if default_ssl_context:
                hostname_checks_common_name = False
            else:
                hostname_checks_common_name = getattr(context, 'hostname_checks_common_name', False) or False
            _match_hostname(cert, assert_hostname or server_hostname, hostname_checks_common_name)
        return _WrappedAndVerifiedSocket(socket=ssl_sock, is_verified=context.verify_mode == ssl.CERT_REQUIRED or bool(assert_fingerprint))
    except BaseException:
        ssl_sock.close()
        raise