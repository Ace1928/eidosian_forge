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
class RSAAlgorithm(Algorithm):
    """
        Performs signing and verification operations using
        RSASSA-PKCS-v1_5 and the specified hash function.
        """
    SHA256: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA256
    SHA384: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA384
    SHA512: ClassVar[type[hashes.HashAlgorithm]] = hashes.SHA512

    def __init__(self, hash_alg: type[hashes.HashAlgorithm]) -> None:
        self.hash_alg = hash_alg

    def prepare_key(self, key: AllowedRSAKeys | str | bytes) -> AllowedRSAKeys:
        if isinstance(key, (RSAPrivateKey, RSAPublicKey)):
            return key
        if not isinstance(key, (bytes, str)):
            raise TypeError('Expecting a PEM-formatted key.')
        key_bytes = force_bytes(key)
        try:
            if key_bytes.startswith(b'ssh-rsa'):
                return cast(RSAPublicKey, load_ssh_public_key(key_bytes))
            else:
                return cast(RSAPrivateKey, load_pem_private_key(key_bytes, password=None))
        except ValueError:
            return cast(RSAPublicKey, load_pem_public_key(key_bytes))

    @overload
    @staticmethod
    def to_jwk(key_obj: AllowedRSAKeys, as_dict: Literal[True]) -> JWKDict:
        ...

    @overload
    @staticmethod
    def to_jwk(key_obj: AllowedRSAKeys, as_dict: Literal[False]=False) -> str:
        ...

    @staticmethod
    def to_jwk(key_obj: AllowedRSAKeys, as_dict: bool=False) -> Union[JWKDict, str]:
        obj: dict[str, Any] | None = None
        if hasattr(key_obj, 'private_numbers'):
            numbers = key_obj.private_numbers()
            obj = {'kty': 'RSA', 'key_ops': ['sign'], 'n': to_base64url_uint(numbers.public_numbers.n).decode(), 'e': to_base64url_uint(numbers.public_numbers.e).decode(), 'd': to_base64url_uint(numbers.d).decode(), 'p': to_base64url_uint(numbers.p).decode(), 'q': to_base64url_uint(numbers.q).decode(), 'dp': to_base64url_uint(numbers.dmp1).decode(), 'dq': to_base64url_uint(numbers.dmq1).decode(), 'qi': to_base64url_uint(numbers.iqmp).decode()}
        elif hasattr(key_obj, 'verify'):
            numbers = key_obj.public_numbers()
            obj = {'kty': 'RSA', 'key_ops': ['verify'], 'n': to_base64url_uint(numbers.n).decode(), 'e': to_base64url_uint(numbers.e).decode()}
        else:
            raise InvalidKeyError('Not a public or private key')
        if as_dict:
            return obj
        else:
            return json.dumps(obj)

    @staticmethod
    def from_jwk(jwk: str | JWKDict) -> AllowedRSAKeys:
        try:
            if isinstance(jwk, str):
                obj = json.loads(jwk)
            elif isinstance(jwk, dict):
                obj = jwk
            else:
                raise ValueError
        except ValueError:
            raise InvalidKeyError('Key is not valid JSON')
        if obj.get('kty') != 'RSA':
            raise InvalidKeyError('Not an RSA key')
        if 'd' in obj and 'e' in obj and ('n' in obj):
            if 'oth' in obj:
                raise InvalidKeyError('Unsupported RSA private key: > 2 primes not supported')
            other_props = ['p', 'q', 'dp', 'dq', 'qi']
            props_found = [prop in obj for prop in other_props]
            any_props_found = any(props_found)
            if any_props_found and (not all(props_found)):
                raise InvalidKeyError('RSA key must include all parameters if any are present besides d')
            public_numbers = RSAPublicNumbers(from_base64url_uint(obj['e']), from_base64url_uint(obj['n']))
            if any_props_found:
                numbers = RSAPrivateNumbers(d=from_base64url_uint(obj['d']), p=from_base64url_uint(obj['p']), q=from_base64url_uint(obj['q']), dmp1=from_base64url_uint(obj['dp']), dmq1=from_base64url_uint(obj['dq']), iqmp=from_base64url_uint(obj['qi']), public_numbers=public_numbers)
            else:
                d = from_base64url_uint(obj['d'])
                p, q = rsa_recover_prime_factors(public_numbers.n, d, public_numbers.e)
                numbers = RSAPrivateNumbers(d=d, p=p, q=q, dmp1=rsa_crt_dmp1(d, p), dmq1=rsa_crt_dmq1(d, q), iqmp=rsa_crt_iqmp(p, q), public_numbers=public_numbers)
            return numbers.private_key()
        elif 'n' in obj and 'e' in obj:
            return RSAPublicNumbers(from_base64url_uint(obj['e']), from_base64url_uint(obj['n'])).public_key()
        else:
            raise InvalidKeyError('Not a public or private key')

    def sign(self, msg: bytes, key: RSAPrivateKey) -> bytes:
        return key.sign(msg, padding.PKCS1v15(), self.hash_alg())

    def verify(self, msg: bytes, key: RSAPublicKey, sig: bytes) -> bool:
        try:
            key.verify(sig, msg, padding.PKCS1v15(), self.hash_alg())
            return True
        except InvalidSignature:
            return False