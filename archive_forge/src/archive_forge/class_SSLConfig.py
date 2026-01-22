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
class SSLConfig:
    """
    SSL Configuration.
    """
    DEFAULT_CA_BUNDLE_PATH = Path(certifi.where())

    def __init__(self, *, cert: CertTypes | None=None, verify: VerifyTypes=True, trust_env: bool=True, http2: bool=False) -> None:
        self.cert = cert
        self.verify = verify
        self.trust_env = trust_env
        self.http2 = http2
        self.ssl_context = self.load_ssl_context()

    def load_ssl_context(self) -> ssl.SSLContext:
        logger.debug('load_ssl_context verify=%r cert=%r trust_env=%r http2=%r', self.verify, self.cert, self.trust_env, self.http2)
        if self.verify:
            return self.load_ssl_context_verify()
        return self.load_ssl_context_no_verify()

    def load_ssl_context_no_verify(self) -> ssl.SSLContext:
        """
        Return an SSL context for unverified connections.
        """
        context = self._create_default_ssl_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        self._load_client_certs(context)
        return context

    def load_ssl_context_verify(self) -> ssl.SSLContext:
        """
        Return an SSL context for verified connections.
        """
        if self.trust_env and self.verify is True:
            ca_bundle = get_ca_bundle_from_env()
            if ca_bundle is not None:
                self.verify = ca_bundle
        if isinstance(self.verify, ssl.SSLContext):
            context = self.verify
            self._load_client_certs(context)
            return context
        elif isinstance(self.verify, bool):
            ca_bundle_path = self.DEFAULT_CA_BUNDLE_PATH
        elif Path(self.verify).exists():
            ca_bundle_path = Path(self.verify)
        else:
            raise IOError('Could not find a suitable TLS CA certificate bundle, invalid path: {}'.format(self.verify))
        context = self._create_default_ssl_context()
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        try:
            context.post_handshake_auth = True
        except AttributeError:
            pass
        try:
            context.hostname_checks_common_name = False
        except AttributeError:
            pass
        if ca_bundle_path.is_file():
            cafile = str(ca_bundle_path)
            logger.debug('load_verify_locations cafile=%r', cafile)
            context.load_verify_locations(cafile=cafile)
        elif ca_bundle_path.is_dir():
            capath = str(ca_bundle_path)
            logger.debug('load_verify_locations capath=%r', capath)
            context.load_verify_locations(capath=capath)
        self._load_client_certs(context)
        return context

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

    def _load_client_certs(self, ssl_context: ssl.SSLContext) -> None:
        """
        Loads client certificates into our SSLContext object
        """
        if self.cert is not None:
            if isinstance(self.cert, str):
                ssl_context.load_cert_chain(certfile=self.cert)
            elif isinstance(self.cert, tuple) and len(self.cert) == 2:
                ssl_context.load_cert_chain(certfile=self.cert[0], keyfile=self.cert[1])
            elif isinstance(self.cert, tuple) and len(self.cert) == 3:
                ssl_context.load_cert_chain(certfile=self.cert[0], keyfile=self.cert[1], password=self.cert[2])