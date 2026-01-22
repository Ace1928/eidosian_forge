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
def _init_cipher(ciphername: bytes, password: typing.Optional[bytes], salt: bytes, rounds: int) -> Cipher[typing.Union[modes.CBC, modes.CTR, modes.GCM]]:
    """Generate key + iv and return cipher."""
    if not password:
        raise ValueError('Key is password-protected.')
    ciph = _SSH_CIPHERS[ciphername]
    seed = _bcrypt_kdf(password, salt, ciph.key_len + ciph.iv_len, rounds, True)
    return Cipher(ciph.alg(seed[:ciph.key_len]), ciph.mode(seed[ciph.key_len:]))