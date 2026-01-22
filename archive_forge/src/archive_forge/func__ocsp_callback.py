from __future__ import annotations
import logging as _logging
import re as _re
from datetime import datetime as _datetime
from datetime import timezone
from typing import TYPE_CHECKING, Iterable, Optional, Type, Union
from cryptography.exceptions import InvalidSignature as _InvalidSignature
from cryptography.hazmat.backends import default_backend as _default_backend
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey as _DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA as _ECDSA
from cryptography.hazmat.primitives.asymmetric.ec import (
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15 as _PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey as _RSAPublicKey
from cryptography.hazmat.primitives.asymmetric.x448 import (
from cryptography.hazmat.primitives.asymmetric.x25519 import (
from cryptography.hazmat.primitives.hashes import SHA1 as _SHA1
from cryptography.hazmat.primitives.hashes import Hash as _Hash
from cryptography.hazmat.primitives.serialization import Encoding as _Encoding
from cryptography.hazmat.primitives.serialization import PublicFormat as _PublicFormat
from cryptography.x509 import AuthorityInformationAccess as _AuthorityInformationAccess
from cryptography.x509 import ExtendedKeyUsage as _ExtendedKeyUsage
from cryptography.x509 import ExtensionNotFound as _ExtensionNotFound
from cryptography.x509 import TLSFeature as _TLSFeature
from cryptography.x509 import TLSFeatureType as _TLSFeatureType
from cryptography.x509 import load_pem_x509_certificate as _load_pem_x509_certificate
from cryptography.x509.ocsp import OCSPCertStatus as _OCSPCertStatus
from cryptography.x509.ocsp import OCSPRequestBuilder as _OCSPRequestBuilder
from cryptography.x509.ocsp import OCSPResponseStatus as _OCSPResponseStatus
from cryptography.x509.ocsp import load_der_ocsp_response as _load_der_ocsp_response
from cryptography.x509.oid import (
from cryptography.x509.oid import ExtendedKeyUsageOID as _ExtendedKeyUsageOID
from requests import post as _post
from requests.exceptions import RequestException as _RequestException
from pymongo import _csot
def _ocsp_callback(conn: Connection, ocsp_bytes: bytes, user_data: Optional[_CallbackData]) -> bool:
    """Callback for use with OpenSSL.SSL.Context.set_ocsp_client_callback."""
    assert user_data
    pycert = conn.get_peer_certificate()
    if pycert is None:
        _LOGGER.debug('No peer cert?')
        return False
    cert = pycert.to_cryptography()
    if hasattr(conn, 'get_verified_chain'):
        pychain = conn.get_verified_chain()
        trusted_ca_certs = None
    else:
        pychain = conn.get_peer_cert_chain()
        trusted_ca_certs = user_data.trusted_ca_certs
    if not pychain:
        _LOGGER.debug('No peer cert chain?')
        return False
    chain = [cer.to_cryptography() for cer in pychain]
    issuer = _get_issuer_cert(cert, chain, trusted_ca_certs)
    must_staple = False
    ext_tls = _get_extension(cert, _TLSFeature)
    if ext_tls is not None:
        for feature in ext_tls.value:
            if feature == _TLSFeatureType.status_request:
                _LOGGER.debug('Peer presented a must-staple cert')
                must_staple = True
                break
    ocsp_response_cache = user_data.ocsp_response_cache
    if ocsp_bytes == b'':
        _LOGGER.debug('Peer did not staple an OCSP response')
        if must_staple:
            _LOGGER.debug('Must-staple cert with no stapled response, hard fail.')
            return False
        if not user_data.check_ocsp_endpoint:
            _LOGGER.debug('OCSP endpoint checking is disabled, soft fail.')
            return True
        ext_aia = _get_extension(cert, _AuthorityInformationAccess)
        if ext_aia is None:
            _LOGGER.debug('No authority access information, soft fail')
            return True
        uris = [desc.access_location.value for desc in ext_aia.value if desc.access_method == _AuthorityInformationAccessOID.OCSP]
        if not uris:
            _LOGGER.debug('No OCSP URI, soft fail')
            return True
        if issuer is None:
            _LOGGER.debug('No issuer cert?')
            return False
        _LOGGER.debug('Requesting OCSP data')
        for uri in uris:
            _LOGGER.debug('Trying %s', uri)
            response = _get_ocsp_response(cert, issuer, uri, ocsp_response_cache)
            if response is None:
                continue
            _LOGGER.debug('OCSP cert status: %r', response.certificate_status)
            if response.certificate_status == _OCSPCertStatus.GOOD:
                return True
            if response.certificate_status == _OCSPCertStatus.REVOKED:
                return False
        _LOGGER.debug('No definitive OCSP cert status, soft fail')
        return True
    _LOGGER.debug('Peer stapled an OCSP response')
    if issuer is None:
        _LOGGER.debug('No issuer cert?')
        return False
    response = _load_der_ocsp_response(ocsp_bytes)
    _LOGGER.debug('OCSP response status: %r', response.response_status)
    if response.response_status != _OCSPResponseStatus.SUCCESSFUL:
        return False
    if not _verify_response(issuer, response):
        return False
    ocsp_response_cache[_build_ocsp_request(cert, issuer)] = response
    _LOGGER.debug('OCSP cert status: %r', response.certificate_status)
    if response.certificate_status == _OCSPCertStatus.REVOKED:
        return False
    return True