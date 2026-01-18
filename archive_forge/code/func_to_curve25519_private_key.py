from typing import Optional
import nacl.bindings
from nacl import encoding
from nacl import exceptions as exc
from nacl.public import (
from nacl.utils import StringFixer, random
def to_curve25519_private_key(self) -> _Curve25519_PrivateKey:
    """
        Converts a :class:`~nacl.signing.SigningKey` to a
        :class:`~nacl.public.PrivateKey`

        :rtype: :class:`~nacl.public.PrivateKey`
        """
    sk = self._signing_key
    raw_private = nacl.bindings.crypto_sign_ed25519_sk_to_curve25519(sk)
    return _Curve25519_PrivateKey(raw_private)