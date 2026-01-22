from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA512_224(HashAlgorithm):
    name = 'sha512-224'
    digest_size = 28
    block_size = 128