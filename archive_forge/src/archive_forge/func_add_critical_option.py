from __future__ import annotations
import binascii
import enum
import os
import re
import typing
import warnings
from base64 import encodebytes as _base64_encode
from dataclasses import dataclass
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.ciphers import (
from cryptography.hazmat.primitives.serialization import (
def add_critical_option(self, name: bytes, value: bytes) -> SSHCertificateBuilder:
    if not isinstance(name, bytes) or not isinstance(value, bytes):
        raise TypeError('name and value must be bytes')
    if name in [name for name, _ in self._critical_options]:
        raise ValueError('Duplicate critical option name')
    return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options + [(name, value)], _extensions=self._extensions)