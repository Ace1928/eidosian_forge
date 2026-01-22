from __future__ import annotations
import abc
import ipaddress
import typing
from email.utils import parseaddr
from cryptography.x509.name import Name
from cryptography.x509.oid import ObjectIdentifier
class DirectoryName(GeneralName):

    def __init__(self, value: Name) -> None:
        if not isinstance(value, Name):
            raise TypeError('value must be a Name')
        self._value = value

    @property
    def value(self) -> Name:
        return self._value

    def __repr__(self) -> str:
        return f'<DirectoryName(value={self.value})>'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectoryName):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)