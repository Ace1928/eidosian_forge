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
def _allowedKeyExchangeMessageType(self, messageType):
    """
        Determine if the given message type may be sent while key exchange is
        in progress.

        @param messageType: The type of message
        @type messageType: L{int}

        @return: C{True} if the given type of message may be sent while key
            exchange is in progress, C{False} if it may not.
        @rtype: L{bool}

        @see: U{http://tools.ietf.org/html/rfc4253#section-7.1}
        """
    if 1 <= messageType <= 19:
        return messageType not in (MSG_SERVICE_REQUEST, MSG_SERVICE_ACCEPT, MSG_EXT_INFO)
    if 20 <= messageType <= 29:
        return messageType not in (MSG_KEXINIT,)
    return 30 <= messageType <= 49