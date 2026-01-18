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
def verify_ip_address(connection: Connection, ip_address: str) -> None:
    """
    Verify whether the certificate of *connection* is valid for *ip_address*.

    Args:
        connection: A pyOpenSSL connection object.

        ip_address:
            The IP address that *connection* should be connected to. Can be an
            IPv4 or IPv6 address.

    Raises:
        service_identity.VerificationError:
            If *connection* does not provide a certificate that is valid for
            *ip_address*.

        service_identity.CertificateError:
            If the certificate chain of *connection* contains a certificate
            that contains invalid/unexpected data.

    .. versionadded:: 18.1.0

    .. versionchanged:: 24.1.0
        :exc:`~service_identity.CertificateError` is raised if the certificate
        contains no ``subjectAltName``\\ s instead of
        :exc:`~service_identity.VerificationError`.
    """
    verify_service_identity(cert_patterns=extract_patterns(connection.get_peer_certificate()), obligatory_ids=[IPAddress_ID(ip_address)], optional_ids=[])