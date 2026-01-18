import base64
from contextlib import closing
import gzip
from http.server import BaseHTTPRequestHandler
import os
import socket
from socketserver import ThreadingMixIn
import ssl
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib.request import (
from wsgiref.simple_server import make_server, WSGIRequestHandler, WSGIServer
from .openmetrics import exposition as openmetrics
from .registry import CollectorRegistry, REGISTRY
from .utils import floatToGoString
from .asgi import make_asgi_app  # noqa
def tls_auth_handler(url: str, method: str, timeout: Optional[float], headers: List[Tuple[str, str]], data: bytes, certfile: str, keyfile: str, cafile: Optional[str]=None, protocol: int=ssl.PROTOCOL_TLS_CLIENT, insecure_skip_verify: bool=False) -> Callable[[], None]:
    """Handler that implements an HTTPS connection with TLS Auth.

    The default protocol (ssl.PROTOCOL_TLS_CLIENT) will also enable
    ssl.CERT_REQUIRED and SSLContext.check_hostname by default. This can be
    disabled by setting insecure_skip_verify to True.

    Both this handler and the TLS feature on pushgateay are experimental."""
    context = ssl.SSLContext(protocol=protocol)
    if cafile is not None:
        context.load_verify_locations(cafile)
    else:
        context.load_default_certs()
    if insecure_skip_verify:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    handler = HTTPSHandler(context=context)
    return _make_handler(url, method, timeout, headers, data, handler)