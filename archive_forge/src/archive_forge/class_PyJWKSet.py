from __future__ import annotations
import json
import time
from typing import Any
from .algorithms import get_default_algorithms, has_crypto, requires_cryptography
from .exceptions import InvalidKeyError, PyJWKError, PyJWKSetError, PyJWTError
from .types import JWKDict
class PyJWKSet:

    def __init__(self, keys: list[JWKDict]) -> None:
        self.keys = []
        if not keys:
            raise PyJWKSetError('The JWK Set did not contain any keys')
        if not isinstance(keys, list):
            raise PyJWKSetError('Invalid JWK Set value')
        for key in keys:
            try:
                self.keys.append(PyJWK(key))
            except PyJWTError:
                continue
        if len(self.keys) == 0:
            raise PyJWKSetError("The JWK Set did not contain any usable keys. Perhaps 'cryptography' is not installed?")

    @staticmethod
    def from_dict(obj: dict[str, Any]) -> 'PyJWKSet':
        keys = obj.get('keys', [])
        return PyJWKSet(keys)

    @staticmethod
    def from_json(data: str) -> 'PyJWKSet':
        obj = json.loads(data)
        return PyJWKSet.from_dict(obj)

    def __getitem__(self, kid: str) -> 'PyJWK':
        for key in self.keys:
            if key.key_id == kid:
                return key
        raise KeyError(f'keyset has no key for kid: {kid}')