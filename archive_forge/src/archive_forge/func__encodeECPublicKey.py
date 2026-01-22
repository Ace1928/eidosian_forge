from __future__ import annotations
import binascii
import hmac
import struct
import types
import zlib
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Any, Callable, Dict, Tuple, Union
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import dh, ec, x25519
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from typing_extensions import Literal
from twisted import __version__ as twisted_version
from twisted.conch.ssh import _kex, address, keys
from twisted.conch.ssh.common import MP, NS, ffs, getMP, getNS
from twisted.internet import defer, protocol
from twisted.logger import Logger
from twisted.python import randbytes
from twisted.python.compat import iterbytes, networkString
def _encodeECPublicKey(self, ecPub):
    """
        Encode an elliptic curve public key to bytes.

        @type ecPub: The appropriate public key type matching
            C{self.kexAlg}: L{ec.EllipticCurvePublicKey} for
            C{ecdh-sha2-nistp*}, or L{x25519.X25519PublicKey} for
            C{curve25519-sha256}.
        @param ecPub: The public key to encode.

        @rtype: L{bytes}
        @return: The encoded public key.
        """
    if self.kexAlg.startswith(b'ecdh-sha2-nistp'):
        return ecPub.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    elif self.kexAlg in (b'curve25519-sha256', b'curve25519-sha256@libssh.org'):
        return ecPub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    else:
        raise UnsupportedAlgorithm(f'Cannot encode elliptic curve public key for {self.kexAlg!r}')