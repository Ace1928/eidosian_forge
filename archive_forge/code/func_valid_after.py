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
def valid_after(self, valid_after: typing.Union[int, float]) -> SSHCertificateBuilder:
    if not isinstance(valid_after, (int, float)):
        raise TypeError('valid_after must be an int or float')
    valid_after = int(valid_after)
    if valid_after < 0 or valid_after >= 2 ** 64:
        raise ValueError('valid_after must [0, 2**64)')
    if self._valid_after is not None:
        raise ValueError('valid_after already set')
    return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=valid_after, _critical_options=self._critical_options, _extensions=self._extensions)