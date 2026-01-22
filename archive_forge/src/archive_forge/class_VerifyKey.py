from typing import Optional
import nacl.bindings
from nacl import encoding
from nacl import exceptions as exc
from nacl.public import (
from nacl.utils import StringFixer, random
class VerifyKey(encoding.Encodable, StringFixer):
    """
    The public key counterpart to an Ed25519 SigningKey for producing digital
    signatures.

    :param key: [:class:`bytes`] Serialized Ed25519 public key
    :param encoder: A class that is able to decode the `key`
    """

    def __init__(self, key: bytes, encoder: encoding.Encoder=encoding.RawEncoder):
        key = encoder.decode(key)
        if not isinstance(key, bytes):
            raise exc.TypeError('VerifyKey must be created from 32 bytes')
        if len(key) != nacl.bindings.crypto_sign_PUBLICKEYBYTES:
            raise exc.ValueError('The key must be exactly %s bytes long' % nacl.bindings.crypto_sign_PUBLICKEYBYTES)
        self._key = key

    def __bytes__(self) -> bytes:
        return self._key

    def __hash__(self) -> int:
        return hash(bytes(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return nacl.bindings.sodium_memcmp(bytes(self), bytes(other))

    def __ne__(self, other: object) -> bool:
        return not self == other

    def verify(self, smessage: bytes, signature: Optional[bytes]=None, encoder: encoding.Encoder=encoding.RawEncoder) -> bytes:
        """
        Verifies the signature of a signed message, returning the message
        if it has not been tampered with else raising
        :class:`~nacl.signing.BadSignatureError`.

        :param smessage: [:class:`bytes`] Either the original messaged or a
            signature and message concated together.
        :param signature: [:class:`bytes`] If an unsigned message is given for
            smessage then the detached signature must be provided.
        :param encoder: A class that is able to decode the secret message and
            signature.
        :rtype: :class:`bytes`
        """
        if signature is not None:
            if not isinstance(signature, bytes):
                raise exc.TypeError('Verification signature must be created from %d bytes' % nacl.bindings.crypto_sign_BYTES)
            if len(signature) != nacl.bindings.crypto_sign_BYTES:
                raise exc.ValueError('The signature must be exactly %d bytes long' % nacl.bindings.crypto_sign_BYTES)
            smessage = signature + encoder.decode(smessage)
        else:
            smessage = encoder.decode(smessage)
        return nacl.bindings.crypto_sign_open(smessage, self._key)

    def to_curve25519_public_key(self) -> _Curve25519_PublicKey:
        """
        Converts a :class:`~nacl.signing.VerifyKey` to a
        :class:`~nacl.public.PublicKey`

        :rtype: :class:`~nacl.public.PublicKey`
        """
        raw_pk = nacl.bindings.crypto_sign_ed25519_pk_to_curve25519(self._key)
        return _Curve25519_PublicKey(raw_pk)