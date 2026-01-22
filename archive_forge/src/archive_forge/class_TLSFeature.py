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
class TLSFeature(ExtensionType):
    oid = ExtensionOID.TLS_FEATURE

    def __init__(self, features: typing.Iterable[TLSFeatureType]) -> None:
        features = list(features)
        if not all((isinstance(x, TLSFeatureType) for x in features)) or len(features) == 0:
            raise TypeError('features must be a list of elements from the TLSFeatureType enum')
        self._features = features
    __len__, __iter__, __getitem__ = _make_sequence_methods('_features')

    def __repr__(self) -> str:
        return f'<TLSFeature(features={self._features})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TLSFeature):
            return NotImplemented
        return self._features == other._features

    def __hash__(self) -> int:
        return hash(tuple(self._features))

    def public_bytes(self) -> bytes:
        return rust_x509.encode_extension_value(self)