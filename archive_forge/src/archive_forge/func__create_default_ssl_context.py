from __future__ import annotations
import logging
import os
import ssl
import typing
from pathlib import Path
import certifi
from ._compat import set_minimum_tls_version_1_2
from ._models import Headers
from ._types import CertTypes, HeaderTypes, TimeoutTypes, URLTypes, VerifyTypes
from ._urls import URL
from ._utils import get_ca_bundle_from_env
def _create_default_ssl_context(self) -> ssl.SSLContext:
    """
        Creates the default SSLContext object that's used for both verified
        and unverified connections.
        """
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    set_minimum_tls_version_1_2(context)
    context.options |= ssl.OP_NO_COMPRESSION
    context.set_ciphers(DEFAULT_CIPHERS)
    if ssl.HAS_ALPN:
        alpn_idents = ['http/1.1', 'h2'] if self.http2 else ['http/1.1']
        context.set_alpn_protocols(alpn_idents)
    keylogfile = os.environ.get('SSLKEYLOGFILE')
    if keylogfile and self.trust_env:
        context.keylog_filename = keylogfile
    return context