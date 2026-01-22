from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA512_256(HashAlgorithm):
    name = 'sha512-256'
    digest_size = 32
    block_size = 128