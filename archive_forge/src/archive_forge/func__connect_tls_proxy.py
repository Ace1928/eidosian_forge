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
def _connect_tls_proxy(self, hostname, conn):
    """
        Establish a TLS connection to the proxy using the provided SSL context.
        """
    proxy_config = self.proxy_config
    ssl_context = proxy_config.ssl_context
    if ssl_context:
        return ssl_wrap_socket(sock=conn, server_hostname=hostname, ssl_context=ssl_context)
    ssl_context = create_proxy_ssl_context(self.ssl_version, self.cert_reqs, self.ca_certs, self.ca_cert_dir, self.ca_cert_data)
    socket = ssl_wrap_socket(sock=conn, ca_certs=self.ca_certs, ca_cert_dir=self.ca_cert_dir, ca_cert_data=self.ca_cert_data, server_hostname=hostname, ssl_context=ssl_context)
    if ssl_context.verify_mode != ssl.CERT_NONE and (not getattr(ssl_context, 'check_hostname', False)):
        cert = socket.getpeercert()
        if not cert.get('subjectAltName', ()):
            warnings.warn('Certificate for {0} has no `subjectAltName`, falling back to check for a `commonName` for now. This feature is being removed by major browsers and deprecated by RFC 2818. (See https://github.com/urllib3/urllib3/issues/497 for details.)'.format(hostname), SubjectAltNameWarning)
        _match_hostname(cert, hostname)
    self.proxy_is_verified = ssl_context.verify_mode == ssl.CERT_REQUIRED
    return socket