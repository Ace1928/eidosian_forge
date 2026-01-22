from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
class MD5(HashAlgorithm):
    name = 'md5'
    digest_size = 16
    block_size = 64