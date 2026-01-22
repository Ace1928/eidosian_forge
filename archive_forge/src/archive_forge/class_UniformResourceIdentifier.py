from __future__ import annotations
import abc
import ipaddress
import typing
from email.utils import parseaddr
from cryptography.x509.name import Name
from cryptography.x509.oid import ObjectIdentifier
class UniformResourceIdentifier(GeneralName):

    def __init__(self, value: str) -> None:
        if isinstance(value, str):
            try:
                value.encode('ascii')
            except UnicodeEncodeError:
                raise ValueError('URI values should be passed as an A-label string. This means unicode characters should be encoded via a library like idna.')
        else:
            raise TypeError('value must be string')
        self._value = value

    @property
    def value(self) -> str:
        return self._value

    @classmethod
    def _init_without_validation(cls, value: str) -> UniformResourceIdentifier:
        instance = cls.__new__(cls)
        instance._value = value
        return instance

    def __repr__(self) -> str:
        return f'<UniformResourceIdentifier(value={self.value!r})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UniformResourceIdentifier):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)