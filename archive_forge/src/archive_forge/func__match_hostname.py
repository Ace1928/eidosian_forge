from __future__ import absolute_import
import datetime
import logging
import os
import re
import socket
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .packages import six
from .packages.six.moves.http_client import HTTPConnection as _HTTPConnection
from .packages.six.moves.http_client import HTTPException  # noqa: F401
from .util.proxy import create_proxy_ssl_context
from ._collections import HTTPHeaderDict  # noqa (historical, removed in v2)
from ._version import __version__
from .exceptions import (
from .util import SKIP_HEADER, SKIPPABLE_HEADERS, connection
from .util.ssl_ import (
from .util.ssl_match_hostname import CertificateError, match_hostname
def _match_hostname(cert, asserted_hostname):
    stripped_hostname = asserted_hostname.strip('u[]')
    if is_ipaddress(stripped_hostname):
        asserted_hostname = stripped_hostname
    try:
        match_hostname(cert, asserted_hostname)
    except CertificateError as e:
        log.warning('Certificate did not match expected hostname: %s. Certificate: %s', asserted_hostname, cert)
        e._peer_cert = cert
        raise