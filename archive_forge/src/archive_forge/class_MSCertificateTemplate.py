from __future__ import annotations
import abc
import datetime
import hashlib
import ipaddress
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import asn1
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import constant_time, serialization
from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.certificate_transparency import (
from cryptography.x509.general_name import (
from cryptography.x509.name import Name, RelativeDistinguishedName
from cryptography.x509.oid import (
class MSCertificateTemplate(ExtensionType):
    oid = ExtensionOID.MS_CERTIFICATE_TEMPLATE

    def __init__(self, template_id: ObjectIdentifier, major_version: typing.Optional[int], minor_version: typing.Optional[int]) -> None:
        if not isinstance(template_id, ObjectIdentifier):
            raise TypeError('oid must be an ObjectIdentifier')
        self._template_id = template_id
        if major_version is not None and (not isinstance(major_version, int)) or (minor_version is not None and (not isinstance(minor_version, int))):
            raise TypeError('major_version and minor_version must be integers or None')
        self._major_version = major_version
        self._minor_version = minor_version

    @property
    def template_id(self) -> ObjectIdentifier:
        return self._template_id

    @property
    def major_version(self) -> typing.Optional[int]:
        return self._major_version

    @property
    def minor_version(self) -> typing.Optional[int]:
        return self._minor_version

    def __repr__(self) -> str:
        return f'<MSCertificateTemplate(template_id={self.template_id}, major_version={self.major_version}, minor_version={self.minor_version})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MSCertificateTemplate):
            return NotImplemented
        return self.template_id == other.template_id and self.major_version == other.major_version and (self.minor_version == other.minor_version)

    def __hash__(self) -> int:
        return hash((self.template_id, self.major_version, self.minor_version))

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)