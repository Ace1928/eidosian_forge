from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SHA224(HashAlgorithm):
    name = 'sha224'
    digest_size = 28
    block_size = 64