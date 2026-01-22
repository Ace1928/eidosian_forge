from __future__ import annotations
import binascii
import struct
import unicodedata
import warnings
from base64 import b64encode, decodebytes, encodebytes
from hashlib import md5, sha256
from typing import Any
import bcrypt
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
from typing_extensions import Literal
from twisted.conch.ssh import common, sexpy
from twisted.conch.ssh.common import int_to_bytes
from twisted.python import randbytes
from twisted.python.compat import iterbytes, nativeString
from twisted.python.constants import NamedConstant, Names
from twisted.python.deprecate import _mutuallyExclusiveArguments
@classmethod
def _fromEd25519Components(cls, a, k=None):
    """Build a key from Ed25519 components.

        @param a: The Ed25519 public key, as defined in RFC 8032 section
            5.1.5.
        @type a: L{bytes}

        @param k: The Ed25519 private key, as defined in RFC 8032 section
            5.1.5.
        @type k: L{bytes}
        """
    if Ed25519PublicKey is None or Ed25519PrivateKey is None:
        raise BadKeyError('Ed25519 keys not supported on this system')
    if k is None:
        keyObject = Ed25519PublicKey.from_public_bytes(a)
    else:
        keyObject = Ed25519PrivateKey.from_private_bytes(k)
    return cls(keyObject)