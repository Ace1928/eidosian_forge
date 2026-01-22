from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
@dataclass
class CertificateSecurityState:
    """
    Details about the security state of the page certificate.
    """
    protocol: str
    key_exchange: str
    cipher: str
    certificate: typing.List[str]
    subject_name: str
    issuer: str
    valid_from: network.TimeSinceEpoch
    valid_to: network.TimeSinceEpoch
    certificate_has_weak_signature: bool
    certificate_has_sha1_signature: bool
    modern_ssl: bool
    obsolete_ssl_protocol: bool
    obsolete_ssl_key_exchange: bool
    obsolete_ssl_cipher: bool
    obsolete_ssl_signature: bool
    key_exchange_group: typing.Optional[str] = None
    mac: typing.Optional[str] = None
    certificate_network_error: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificate'] = [i for i in self.certificate]
        json['subjectName'] = self.subject_name
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['certificateHasWeakSignature'] = self.certificate_has_weak_signature
        json['certificateHasSha1Signature'] = self.certificate_has_sha1_signature
        json['modernSSL'] = self.modern_ssl
        json['obsoleteSslProtocol'] = self.obsolete_ssl_protocol
        json['obsoleteSslKeyExchange'] = self.obsolete_ssl_key_exchange
        json['obsoleteSslCipher'] = self.obsolete_ssl_cipher
        json['obsoleteSslSignature'] = self.obsolete_ssl_signature
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.certificate_network_error is not None:
            json['certificateNetworkError'] = self.certificate_network_error
        return json

    @classmethod
    def from_json(cls, json):
        return cls(protocol=str(json['protocol']), key_exchange=str(json['keyExchange']), cipher=str(json['cipher']), certificate=[str(i) for i in json['certificate']], subject_name=str(json['subjectName']), issuer=str(json['issuer']), valid_from=network.TimeSinceEpoch.from_json(json['validFrom']), valid_to=network.TimeSinceEpoch.from_json(json['validTo']), certificate_has_weak_signature=bool(json['certificateHasWeakSignature']), certificate_has_sha1_signature=bool(json['certificateHasSha1Signature']), modern_ssl=bool(json['modernSSL']), obsolete_ssl_protocol=bool(json['obsoleteSslProtocol']), obsolete_ssl_key_exchange=bool(json['obsoleteSslKeyExchange']), obsolete_ssl_cipher=bool(json['obsoleteSslCipher']), obsolete_ssl_signature=bool(json['obsoleteSslSignature']), key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None, mac=str(json['mac']) if 'mac' in json else None, certificate_network_error=str(json['certificateNetworkError']) if 'certificateNetworkError' in json else None)