from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class SM3(HashAlgorithm):
    name = 'sm3'
    digest_size = 32
    block_size = 64