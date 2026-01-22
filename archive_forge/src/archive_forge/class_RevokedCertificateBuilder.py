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
class RevokedCertificateBuilder:

    def __init__(self, serial_number: typing.Optional[int]=None, revocation_date: typing.Optional[datetime.datetime]=None, extensions: typing.List[Extension[ExtensionType]]=[]):
        self._serial_number = serial_number
        self._revocation_date = revocation_date
        self._extensions = extensions

    def serial_number(self, number: int) -> RevokedCertificateBuilder:
        if not isinstance(number, int):
            raise TypeError('Serial number must be of integral type.')
        if self._serial_number is not None:
            raise ValueError('The serial number may only be set once.')
        if number <= 0:
            raise ValueError('The serial number should be positive')
        if number.bit_length() >= 160:
            raise ValueError('The serial number should not be more than 159 bits.')
        return RevokedCertificateBuilder(number, self._revocation_date, self._extensions)

    def revocation_date(self, time: datetime.datetime) -> RevokedCertificateBuilder:
        if not isinstance(time, datetime.datetime):
            raise TypeError('Expecting datetime object.')
        if self._revocation_date is not None:
            raise ValueError('The revocation date may only be set once.')
        time = _convert_to_naive_utc_time(time)
        if time < _EARLIEST_UTC_TIME:
            raise ValueError('The revocation date must be on or after 1950 January 1.')
        return RevokedCertificateBuilder(self._serial_number, time, self._extensions)

    def add_extension(self, extval: ExtensionType, critical: bool) -> RevokedCertificateBuilder:
        if not isinstance(extval, ExtensionType):
            raise TypeError('extension must be an ExtensionType')
        extension = Extension(extval.oid, critical, extval)
        _reject_duplicate_extension(extension, self._extensions)
        return RevokedCertificateBuilder(self._serial_number, self._revocation_date, self._extensions + [extension])

    def build(self, backend: typing.Any=None) -> RevokedCertificate:
        if self._serial_number is None:
            raise ValueError('A revoked certificate must have a serial number')
        if self._revocation_date is None:
            raise ValueError('A revoked certificate must have a revocation date')
        return _RawRevokedCertificate(self._serial_number, self._revocation_date, Extensions(self._extensions))