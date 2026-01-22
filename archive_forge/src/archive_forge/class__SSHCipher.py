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
@dataclass
class _SSHCipher:
    alg: typing.Type[algorithms.AES]
    key_len: int
    mode: typing.Union[typing.Type[modes.CTR], typing.Type[modes.CBC], typing.Type[modes.GCM]]
    block_len: int
    iv_len: int
    tag_len: typing.Optional[int]
    is_aead: bool