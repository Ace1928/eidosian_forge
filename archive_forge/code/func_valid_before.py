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
def valid_before(self, valid_before: typing.Union[int, float]) -> SSHCertificateBuilder:
    if not isinstance(valid_before, (int, float)):
        raise TypeError('valid_before must be an int or float')
    valid_before = int(valid_before)
    if valid_before < 0 or valid_before >= 2 ** 64:
        raise ValueError('valid_before must [0, 2**64)')
    if self._valid_before is not None:
        raise ValueError('valid_before already set')
    return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=self._valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)