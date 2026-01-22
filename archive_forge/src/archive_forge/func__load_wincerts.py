from __future__ import annotations
import socket as _socket
import ssl as _stdlibssl
import sys as _sys
import time as _time
from errno import EINTR as _EINTR
from ipaddress import ip_address as _ip_address
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union
from cryptography.x509 import load_der_x509_certificate as _load_der_x509_certificate
from OpenSSL import SSL as _SSL
from OpenSSL import crypto as _crypto
from service_identity import CertificateError as _SICertificateError
from service_identity import VerificationError as _SIVerificationError
from service_identity.pyopenssl import verify_hostname as _verify_hostname
from service_identity.pyopenssl import verify_ip_address as _verify_ip_address
from pymongo.errors import ConfigurationError as _ConfigurationError
from pymongo.errors import _CertificateError
from pymongo.ocsp_cache import _OCSPCache
from pymongo.ocsp_support import _load_trusted_ca_certs, _ocsp_callback
from pymongo.socket_checker import SocketChecker as _SocketChecker
from pymongo.socket_checker import _errno_from_exception
from pymongo.write_concern import validate_boolean
def _load_wincerts(self, store: str) -> None:
    """Attempt to load CA certs from Windows trust store."""
    cert_store = self._ctx.get_cert_store()
    oid = _stdlibssl.Purpose.SERVER_AUTH.oid
    for cert, encoding, trust in _stdlibssl.enum_certificates(store):
        if encoding == 'x509_asn':
            if trust is True or oid in trust:
                cert_store.add_cert(_crypto.X509.from_cryptography(_load_der_x509_certificate(cert)))