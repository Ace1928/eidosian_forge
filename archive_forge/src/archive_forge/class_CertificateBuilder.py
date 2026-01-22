from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
class CertificateBuilder:
    _extensions: typing.List[Extension[ExtensionType]]

    def __init__(self, issuer_name: typing.Optional[Name]=None, subject_name: typing.Optional[Name]=None, public_key: typing.Optional[CertificatePublicKeyTypes]=None, serial_number: typing.Optional[int]=None, not_valid_before: typing.Optional[datetime.datetime]=None, not_valid_after: typing.Optional[datetime.datetime]=None, extensions: typing.List[Extension[ExtensionType]]=[]) -> None:
        self._version = Version.v3
        self._issuer_name = issuer_name
        self._subject_name = subject_name
        self._public_key = public_key
        self._serial_number = serial_number
        self._not_valid_before = not_valid_before
        self._not_valid_after = not_valid_after
        self._extensions = extensions

    def issuer_name(self, name: Name) -> CertificateBuilder:
        """
        Sets the CA's distinguished name.
        """
        if not isinstance(name, Name):
            raise TypeError('Expecting x509.Name object.')
        if self._issuer_name is not None:
            raise ValueError('The issuer name may only be set once.')
        return CertificateBuilder(name, self._subject_name, self._public_key, self._serial_number, self._not_valid_before, self._not_valid_after, self._extensions)

    def subject_name(self, name: Name) -> CertificateBuilder:
        """
        Sets the requestor's distinguished name.
        """
        if not isinstance(name, Name):
            raise TypeError('Expecting x509.Name object.')
        if self._subject_name is not None:
            raise ValueError('The subject name may only be set once.')
        return CertificateBuilder(self._issuer_name, name, self._public_key, self._serial_number, self._not_valid_before, self._not_valid_after, self._extensions)

    def public_key(self, key: CertificatePublicKeyTypes) -> CertificateBuilder:
        """
        Sets the requestor's public key (as found in the signing request).
        """
        if not isinstance(key, (dsa.DSAPublicKey, rsa.RSAPublicKey, ec.EllipticCurvePublicKey, ed25519.Ed25519PublicKey, ed448.Ed448PublicKey, x25519.X25519PublicKey, x448.X448PublicKey)):
            raise TypeError('Expecting one of DSAPublicKey, RSAPublicKey, EllipticCurvePublicKey, Ed25519PublicKey, Ed448PublicKey, X25519PublicKey, or X448PublicKey.')
        if self._public_key is not None:
            raise ValueError('The public key may only be set once.')
        return CertificateBuilder(self._issuer_name, self._subject_name, key, self._serial_number, self._not_valid_before, self._not_valid_after, self._extensions)

    def serial_number(self, number: int) -> CertificateBuilder:
        """
        Sets the certificate serial number.
        """
        if not isinstance(number, int):
            raise TypeError('Serial number must be of integral type.')
        if self._serial_number is not None:
            raise ValueError('The serial number may only be set once.')
        if number <= 0:
            raise ValueError('The serial number should be positive.')
        if number.bit_length() >= 160:
            raise ValueError('The serial number should not be more than 159 bits.')
        return CertificateBuilder(self._issuer_name, self._subject_name, self._public_key, number, self._not_valid_before, self._not_valid_after, self._extensions)

    def not_valid_before(self, time: datetime.datetime) -> CertificateBuilder:
        """
        Sets the certificate activation time.
        """
        if not isinstance(time, datetime.datetime):
            raise TypeError('Expecting datetime object.')
        if self._not_valid_before is not None:
            raise ValueError('The not valid before may only be set once.')
        time = _convert_to_naive_utc_time(time)
        if time < _EARLIEST_UTC_TIME:
            raise ValueError('The not valid before date must be on or after 1950 January 1).')
        if self._not_valid_after is not None and time > self._not_valid_after:
            raise ValueError('The not valid before date must be before the not valid after date.')
        return CertificateBuilder(self._issuer_name, self._subject_name, self._public_key, self._serial_number, time, self._not_valid_after, self._extensions)

    def not_valid_after(self, time: datetime.datetime) -> CertificateBuilder:
        """
        Sets the certificate expiration time.
        """
        if not isinstance(time, datetime.datetime):
            raise TypeError('Expecting datetime object.')
        if self._not_valid_after is not None:
            raise ValueError('The not valid after may only be set once.')
        time = _convert_to_naive_utc_time(time)
        if time < _EARLIEST_UTC_TIME:
            raise ValueError('The not valid after date must be on or after 1950 January 1.')
        if self._not_valid_before is not None and time < self._not_valid_before:
            raise ValueError('The not valid after date must be after the not valid before date.')
        return CertificateBuilder(self._issuer_name, self._subject_name, self._public_key, self._serial_number, self._not_valid_before, time, self._extensions)

    def add_extension(self, extval: ExtensionType, critical: bool) -> CertificateBuilder:
        """
        Adds an X.509 extension to the certificate.
        """
        if not isinstance(extval, ExtensionType):
            raise TypeError('extension must be an ExtensionType')
        extension = Extension(extval.oid, critical, extval)
        _reject_duplicate_extension(extension, self._extensions)
        return CertificateBuilder(self._issuer_name, self._subject_name, self._public_key, self._serial_number, self._not_valid_before, self._not_valid_after, self._extensions + [extension])

    def sign(self, private_key: CertificateIssuerPrivateKeyTypes, algorithm: typing.Optional[_AllowedHashTypes], backend: typing.Any=None, *, rsa_padding: typing.Optional[typing.Union[padding.PSS, padding.PKCS1v15]]=None) -> Certificate:
        """
        Signs the certificate using the CA's private key.
        """
        if self._subject_name is None:
            raise ValueError('A certificate must have a subject name')
        if self._issuer_name is None:
            raise ValueError('A certificate must have an issuer name')
        if self._serial_number is None:
            raise ValueError('A certificate must have a serial number')
        if self._not_valid_before is None:
            raise ValueError('A certificate must have a not valid before time')
        if self._not_valid_after is None:
            raise ValueError('A certificate must have a not valid after time')
        if self._public_key is None:
            raise ValueError('A certificate must have a public key')
        if rsa_padding is not None:
            if not isinstance(rsa_padding, (padding.PSS, padding.PKCS1v15)):
                raise TypeError('Padding must be PSS or PKCS1v15')
            if not isinstance(private_key, rsa.RSAPrivateKey):
                raise TypeError('Padding is only supported for RSA keys')
        return rust_x509.create_x509_certificate(self, private_key, algorithm, rsa_padding)