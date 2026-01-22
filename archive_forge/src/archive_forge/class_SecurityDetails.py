from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class SecurityDetails:
    """
    Security details about a request.
    """
    protocol: str
    key_exchange: str
    cipher: str
    certificate_id: security.CertificateId
    subject_name: str
    san_list: typing.List[str]
    issuer: str
    valid_from: TimeSinceEpoch
    valid_to: TimeSinceEpoch
    signed_certificate_timestamp_list: typing.List[SignedCertificateTimestamp]
    certificate_transparency_compliance: CertificateTransparencyCompliance
    encrypted_client_hello: bool
    key_exchange_group: typing.Optional[str] = None
    mac: typing.Optional[str] = None
    server_signature_algorithm: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificateId'] = self.certificate_id.to_json()
        json['subjectName'] = self.subject_name
        json['sanList'] = [i for i in self.san_list]
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['signedCertificateTimestampList'] = [i.to_json() for i in self.signed_certificate_timestamp_list]
        json['certificateTransparencyCompliance'] = self.certificate_transparency_compliance.to_json()
        json['encryptedClientHello'] = self.encrypted_client_hello
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.server_signature_algorithm is not None:
            json['serverSignatureAlgorithm'] = self.server_signature_algorithm
        return json

    @classmethod
    def from_json(cls, json):
        return cls(protocol=str(json['protocol']), key_exchange=str(json['keyExchange']), cipher=str(json['cipher']), certificate_id=security.CertificateId.from_json(json['certificateId']), subject_name=str(json['subjectName']), san_list=[str(i) for i in json['sanList']], issuer=str(json['issuer']), valid_from=TimeSinceEpoch.from_json(json['validFrom']), valid_to=TimeSinceEpoch.from_json(json['validTo']), signed_certificate_timestamp_list=[SignedCertificateTimestamp.from_json(i) for i in json['signedCertificateTimestampList']], certificate_transparency_compliance=CertificateTransparencyCompliance.from_json(json['certificateTransparencyCompliance']), encrypted_client_hello=bool(json['encryptedClientHello']), key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None, mac=str(json['mac']) if 'mac' in json else None, server_signature_algorithm=int(json['serverSignatureAlgorithm']) if 'serverSignatureAlgorithm' in json else None)