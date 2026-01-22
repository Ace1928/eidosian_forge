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
def _continueGEX_REPLY(self, ignored, pubKey, f, signature):
    """
        The host key has been verified, so we generate the keys.

        @param ignored: Ignored.

        @param pubKey: the public key blob for the server's public key.
        @type pubKey: L{str}
        @param f: the server's Diffie-Hellman public key.
        @type f: L{int}
        @param signature: the server's signature, verifying that it has the
            correct private key.
        @type signature: L{str}
        """
    serverKey = keys.Key.fromString(pubKey)
    sharedSecret = self._finishEphemeralDH(f)
    h = _kex.getHashProcessor(self.kexAlg)()
    h.update(NS(self.ourVersionString))
    h.update(NS(self.otherVersionString))
    h.update(NS(self.ourKexInitPayload))
    h.update(NS(self.otherKexInitPayload))
    h.update(NS(pubKey))
    h.update(struct.pack('!LLL', self._dhMinimalGroupSize, self._dhPreferredGroupSize, self._dhMaximalGroupSize))
    h.update(MP(self.p))
    h.update(MP(self.g))
    h.update(self.dhSecretKeyPublicMP)
    h.update(MP(f))
    h.update(sharedSecret)
    exchangeHash = h.digest()
    if not serverKey.verify(signature, exchangeHash):
        self.sendDisconnect(DISCONNECT_KEY_EXCHANGE_FAILED, b'bad signature')
        return
    self._keySetup(sharedSecret, exchangeHash)