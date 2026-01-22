import base64
import datetime
import ssl
from urllib.parse import urljoin, urlparse
import cryptography.hazmat.primitives.hashes
import requests
from cryptography import hazmat, x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat import backends
from cryptography.hazmat.primitives.asymmetric.dsa import DSAPublicKey
from cryptography.hazmat.primitives.asymmetric.ec import ECDSA, EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.hashes import SHA1, Hash
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.x509 import ocsp
from redis.exceptions import AuthorizationError, ConnectionError
def _check_certificate(issuer_cert, ocsp_bytes, validate=True):
    """A wrapper the return the validity of a known ocsp certificate"""
    ocsp_response = ocsp.load_der_ocsp_response(ocsp_bytes)
    if ocsp_response.response_status == ocsp.OCSPResponseStatus.UNAUTHORIZED:
        raise AuthorizationError('you are not authorized to view this ocsp certificate')
    if ocsp_response.response_status == ocsp.OCSPResponseStatus.SUCCESSFUL:
        if ocsp_response.certificate_status != ocsp.OCSPCertStatus.GOOD:
            raise ConnectionError(f'Received an {str(ocsp_response.certificate_status).split('.')[1]} ocsp certificate status')
    else:
        raise ConnectionError('failed to retrieve a successful response from the ocsp responder')
    if ocsp_response.this_update >= datetime.datetime.now():
        raise ConnectionError('ocsp certificate was issued in the future')
    if ocsp_response.next_update and ocsp_response.next_update < datetime.datetime.now():
        raise ConnectionError('ocsp certificate has invalid update - in the past')
    responder_name = ocsp_response.responder_name
    issuer_hash = ocsp_response.issuer_key_hash
    responder_hash = ocsp_response.responder_key_hash
    cert_to_validate = issuer_cert
    if responder_name is not None and responder_name == issuer_cert.subject or responder_hash == issuer_hash:
        cert_to_validate = issuer_cert
    else:
        certs = ocsp_response.certificates
        responder_certs = _get_certificates(certs, issuer_cert, responder_name, responder_hash)
        try:
            responder_cert = responder_certs[0]
        except IndexError:
            raise ConnectionError('no certificates found for the responder')
        ext = responder_cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage)
        if ext is None or x509.oid.ExtendedKeyUsageOID.OCSP_SIGNING not in ext.value:
            raise ConnectionError('delegate not autorized for ocsp signing')
        cert_to_validate = responder_cert
    if validate:
        _verify_response(cert_to_validate, ocsp_response)
    return True