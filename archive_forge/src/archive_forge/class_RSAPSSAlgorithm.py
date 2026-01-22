from __future__ import annotations
import hashlib
import hmac
import json
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, Union, cast, overload
from .exceptions import InvalidKeyError
from .types import HashlibHash, JWKDict
from .utils import (
class RSAPSSAlgorithm(RSAAlgorithm):
    """
        Performs a signature using RSASSA-PSS with MGF1
        """

    def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
        return key.sign(msg, padding.PSS(mgf=padding.MGF1(self.hash_alg()), salt_length=self.hash_alg().digest_size), self.hash_alg())

    def verify(self, msg: bytes, key: RSAPublicKey, sig: bytes) -> bool:
        try:
            key.verify(sig, msg, padding.PSS(mgf=padding.MGF1(self.hash_alg()), salt_length=self.hash_alg().digest_size), self.hash_alg())
            return True
        except InvalidSignature:
            return False