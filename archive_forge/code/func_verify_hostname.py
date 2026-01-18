from __future__ import annotations
import contextlib
import warnings
from typing import Sequence
from pyasn1.codec.der.decoder import decode
from pyasn1.type.char import IA5String
from pyasn1.type.univ import ObjectIdentifier
from pyasn1_modules.rfc2459 import GeneralNames
from .exceptions import CertificateError
from .hazmat import (
def verify_hostname(connection: Connection, hostname: str) -> None:
    """
    Verify whether the certificate of *connection* is valid for *hostname*.

    Args:
        connection: A pyOpenSSL connection object.

        hostname: The hostname that *connection* should be connected to.

    Raises:
        service_identity.VerificationError:
            If *connection* does not provide a certificate that is valid for
            *hostname*.

        service_identity.CertificateError:
            If certificate provided by *connection* contains invalid /
            unexpected data. This includes the case where the certificate
            contains no ``subjectAltName``\\ s.

    .. versionchanged:: 24.1.0
        :exc:`~service_identity.CertificateError` is raised if the certificate
        contains no ``subjectAltName``\\ s instead of
        :exc:`~service_identity.VerificationError`.
    """
    verify_service_identity(cert_patterns=extract_patterns(connection.get_peer_certificate()), obligatory_ids=[DNS_ID(hostname)], optional_ids=[])