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
def _getKey(self, c, sharedSecret, exchangeHash):
    """
        Get one of the keys for authentication/encryption.

        @type c: L{bytes}
        @param c: The letter identifying which key this is.

        @type sharedSecret: L{bytes}
        @param sharedSecret: The shared secret K.

        @type exchangeHash: L{bytes}
        @param exchangeHash: The hash H from key exchange.

        @rtype: L{bytes}
        @return: The derived key.
        """
    hashProcessor = _kex.getHashProcessor(self.kexAlg)
    k1 = hashProcessor(sharedSecret + exchangeHash + c + self.sessionID)
    k1 = k1.digest()
    k2 = hashProcessor(sharedSecret + exchangeHash + k1).digest()
    k3 = hashProcessor(sharedSecret + exchangeHash + k1 + k2).digest()
    k4 = hashProcessor(sharedSecret + exchangeHash + k1 + k2 + k3).digest()
    return k1 + k2 + k3 + k4