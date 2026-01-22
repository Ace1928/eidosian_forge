from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
def _wrap_ctx(self, ctx: _BackendCipherContext, encrypt: bool) -> typing.Union[AEADEncryptionContext, AEADDecryptionContext, CipherContext]:
    if isinstance(self.mode, modes.ModeWithAuthenticationTag):
        if encrypt:
            return _AEADEncryptionContext(ctx)
        else:
            return _AEADDecryptionContext(ctx)
    else:
        return _CipherContext(ctx)