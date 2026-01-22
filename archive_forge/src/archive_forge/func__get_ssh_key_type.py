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
def _get_ssh_key_type(key: typing.Union[SSHPrivateKeyTypes, SSHPublicKeyTypes]) -> bytes:
    if isinstance(key, ec.EllipticCurvePrivateKey):
        key_type = _ecdsa_key_type(key.public_key())
    elif isinstance(key, ec.EllipticCurvePublicKey):
        key_type = _ecdsa_key_type(key)
    elif isinstance(key, (rsa.RSAPrivateKey, rsa.RSAPublicKey)):
        key_type = _SSH_RSA
    elif isinstance(key, (dsa.DSAPrivateKey, dsa.DSAPublicKey)):
        key_type = _SSH_DSA
    elif isinstance(key, (ed25519.Ed25519PrivateKey, ed25519.Ed25519PublicKey)):
        key_type = _SSH_ED25519
    else:
        raise ValueError('Unsupported key type')
    return key_type