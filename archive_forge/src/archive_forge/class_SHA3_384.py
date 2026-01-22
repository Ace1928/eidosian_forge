from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA3_384(HashAlgorithm):
    name = 'sha3-384'
    digest_size = 48
    block_size = None