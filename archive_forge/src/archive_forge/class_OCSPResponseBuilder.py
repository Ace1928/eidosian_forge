from __future__ import annotations
import abc
import datetime
import typing
from cryptography import utils, x509
from cryptography.hazmat.bindings._rust import ocsp
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.base import (
class OCSPResponseBuilder:

    def __init__(self, response: typing.Optional[_SingleResponse]=None, responder_id: typing.Optional[typing.Tuple[x509.Certificate, OCSPResponderEncoding]]=None, certs: typing.Optional[typing.List[x509.Certificate]]=None, extensions: typing.List[x509.Extension[x509.ExtensionType]]=[]):
        self._response = response
        self._responder_id = responder_id
        self._certs = certs
        self._extensions = extensions

    def add_response(self, cert: x509.Certificate, issuer: x509.Certificate, algorithm: hashes.HashAlgorithm, cert_status: OCSPCertStatus, this_update: datetime.datetime, next_update: typing.Optional[datetime.datetime], revocation_time: typing.Optional[datetime.datetime], revocation_reason: typing.Optional[x509.ReasonFlags]) -> OCSPResponseBuilder:
        if self._response is not None:
            raise ValueError('Only one response per OCSPResponse.')
        singleresp = _SingleResponse(cert, issuer, algorithm, cert_status, this_update, next_update, revocation_time, revocation_reason)
        return OCSPResponseBuilder(singleresp, self._responder_id, self._certs, self._extensions)

    def responder_id(self, encoding: OCSPResponderEncoding, responder_cert: x509.Certificate) -> OCSPResponseBuilder:
        if self._responder_id is not None:
            raise ValueError('responder_id can only be set once')
        if not isinstance(responder_cert, x509.Certificate):
            raise TypeError('responder_cert must be a Certificate')
        if not isinstance(encoding, OCSPResponderEncoding):
            raise TypeError('encoding must be an element from OCSPResponderEncoding')
        return OCSPResponseBuilder(self._response, (responder_cert, encoding), self._certs, self._extensions)

    def certificates(self, certs: typing.Iterable[x509.Certificate]) -> OCSPResponseBuilder:
        if self._certs is not None:
            raise ValueError('certificates may only be set once')
        certs = list(certs)
        if len(certs) == 0:
            raise ValueError('certs must not be an empty list')
        if not all((isinstance(x, x509.Certificate) for x in certs)):
            raise TypeError('certs must be a list of Certificates')
        return OCSPResponseBuilder(self._response, self._responder_id, certs, self._extensions)

    def add_extension(self, extval: x509.ExtensionType, critical: bool) -> OCSPResponseBuilder:
        if not isinstance(extval, x509.ExtensionType):
            raise TypeError('extension must be an ExtensionType')
        extension = x509.Extension(extval.oid, critical, extval)
        _reject_duplicate_extension(extension, self._extensions)
        return OCSPResponseBuilder(self._response, self._responder_id, self._certs, self._extensions + [extension])

    def sign(self, private_key: CertificateIssuerPrivateKeyTypes, algorithm: typing.Optional[hashes.HashAlgorithm]) -> OCSPResponse:
        if self._response is None:
            raise ValueError('You must add a response before signing')
        if self._responder_id is None:
            raise ValueError('You must add a responder_id before signing')
        return ocsp.create_ocsp_response(OCSPResponseStatus.SUCCESSFUL, self, private_key, algorithm)

    @classmethod
    def build_unsuccessful(cls, response_status: OCSPResponseStatus) -> OCSPResponse:
        if not isinstance(response_status, OCSPResponseStatus):
            raise TypeError('response_status must be an item from OCSPResponseStatus')
        if response_status is OCSPResponseStatus.SUCCESSFUL:
            raise ValueError('response_status cannot be SUCCESSFUL')
        return ocsp.create_ocsp_response(response_status, None, None, None)