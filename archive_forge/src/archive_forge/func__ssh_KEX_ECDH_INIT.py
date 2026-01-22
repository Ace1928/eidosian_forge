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
def _ssh_KEX_ECDH_INIT(self, packet):
    """
        Called from L{ssh_KEX_DH_GEX_REQUEST_OLD} to handle
        elliptic curve key exchanges.

        Payload::

            string client Elliptic Curve Diffie-Hellman public key

        Just like L{_ssh_KEXDH_INIT} this message type is also not dispatched
        directly. Extra check to determine if this is really KEX_ECDH_INIT
        is required.

        First we load the host's public/private keys.
        Then we generate the ECDH public/private keypair for the given curve.
        With that we generate the shared secret key.
        Then we compute the hash to sign and send back to the client
        Along with the server's public key and the ECDH public key.

        @type packet: L{bytes}
        @param packet: The message data.

        @return: None.
        """
    pktPub, packet = getNS(packet)
    pubHostKey, privHostKey = self._getHostKeys(self.keyAlg)
    ecPriv = self._generateECPrivateKey()
    self.ecPub = ecPriv.public_key()
    encPub = self._encodeECPublicKey(self.ecPub)
    sharedSecret = self._generateECSharedSecret(ecPriv, pktPub)
    h = _kex.getHashProcessor(self.kexAlg)()
    h.update(NS(self.otherVersionString))
    h.update(NS(self.ourVersionString))
    h.update(NS(self.otherKexInitPayload))
    h.update(NS(self.ourKexInitPayload))
    h.update(NS(pubHostKey.blob()))
    h.update(NS(pktPub))
    h.update(NS(encPub))
    h.update(sharedSecret)
    exchangeHash = h.digest()
    self.sendPacket(MSG_KEXDH_REPLY, NS(pubHostKey.blob()) + NS(encPub) + NS(privHostKey.sign(exchangeHash, signatureType=self.keyAlg)))
    self._keySetup(sharedSecret, exchangeHash)