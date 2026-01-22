from __future__ import annotations
import os
import typing
from cryptography import exceptions, utils
from cryptography.hazmat.backends.openssl import aead
from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.bindings._rust import FixedPool
def _validate_lengths(self, nonce: bytes, data_len: int) -> None:
    l_val = 15 - len(nonce)
    if 2 ** (8 * l_val) < data_len:
        raise ValueError('Data too long for nonce')