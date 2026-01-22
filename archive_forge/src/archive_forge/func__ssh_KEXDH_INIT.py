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
def _ssh_KEXDH_INIT(self, packet):
    """
        Called to handle the beginning of a non-group key exchange.

        Unlike other message types, this is not dispatched automatically.  It
        is called from C{ssh_KEX_DH_GEX_REQUEST_OLD} because an extra check is
        required to determine if this is really a KEXDH_INIT message or if it
        is a KEX_DH_GEX_REQUEST_OLD message.

        The KEXDH_INIT payload::

                integer e (the client's Diffie-Hellman public key)

        We send the KEXDH_REPLY with our host key and signature.

        @type packet: L{bytes}
        @param packet: The message data.
        """
    clientDHpublicKey, foo = getMP(packet)
    pubHostKey, privHostKey = self._getHostKeys(self.keyAlg)
    self.g, self.p = _kex.getDHGeneratorAndPrime(self.kexAlg)
    self._startEphemeralDH()
    sharedSecret = self._finishEphemeralDH(clientDHpublicKey)
    h = _kex.getHashProcessor(self.kexAlg)()
    h.update(NS(self.otherVersionString))
    h.update(NS(self.ourVersionString))
    h.update(NS(self.otherKexInitPayload))
    h.update(NS(self.ourKexInitPayload))
    h.update(NS(pubHostKey.blob()))
    h.update(MP(clientDHpublicKey))
    h.update(self.dhSecretKeyPublicMP)
    h.update(sharedSecret)
    exchangeHash = h.digest()
    self.sendPacket(MSG_KEXDH_REPLY, NS(pubHostKey.blob()) + self.dhSecretKeyPublicMP + NS(privHostKey.sign(exchangeHash, signatureType=self.keyAlg)))
    self._keySetup(sharedSecret, exchangeHash)