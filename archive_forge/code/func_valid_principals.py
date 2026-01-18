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
def valid_principals(self, valid_principals: typing.List[bytes]) -> SSHCertificateBuilder:
    if self._valid_for_all_principals:
        raise ValueError("Principals can't be set because the cert is valid for all principals")
    if not all((isinstance(x, bytes) for x in valid_principals)) or not valid_principals:
        raise TypeError("principals must be a list of bytes and can't be empty")
    if self._valid_principals:
        raise ValueError('valid_principals already set')
    if len(valid_principals) > _SSHKEY_CERT_MAX_PRINCIPALS:
        raise ValueError('Reached or exceeded the maximum number of valid_principals')
    return SSHCertificateBuilder(_public_key=self._public_key, _serial=self._serial, _type=self._type, _key_id=self._key_id, _valid_principals=valid_principals, _valid_for_all_principals=self._valid_for_all_principals, _valid_before=self._valid_before, _valid_after=self._valid_after, _critical_options=self._critical_options, _extensions=self._extensions)