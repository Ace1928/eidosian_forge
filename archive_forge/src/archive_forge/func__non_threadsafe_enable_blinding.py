from __future__ import annotations
import threading
import typing
from cryptography.exceptions import (
from cryptography.hazmat.backends.openssl.utils import (
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
from cryptography.hazmat.primitives.asymmetric.padding import (
from cryptography.hazmat.primitives.asymmetric.rsa import (
def _non_threadsafe_enable_blinding(self) -> None:
    if not self._blinded:
        res = self._backend._lib.RSA_blinding_on(self._rsa_cdata, self._backend._ffi.NULL)
        self._backend.openssl_assert(res == 1)
        self._blinded = True