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
def _to_mpint(val: int) -> bytes:
    """Storage format for signed bigint."""
    if val < 0:
        raise ValueError('negative mpint not allowed')
    if not val:
        return b''
    nbytes = (val.bit_length() + 8) // 8
    return utils.int_to_bytes(val, nbytes)