from typing import Tuple
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
class crypto_sign_ed25519ph_state:
    """
    State object wrapping the sha-512 state used in ed25519ph computation
    """
    __slots__ = ['state']

    def __init__(self) -> None:
        self.state: bytes = ffi.new('unsigned char[]', crypto_sign_ed25519ph_STATEBYTES)
        rc = lib.crypto_sign_ed25519ph_init(self.state)
        ensure(rc == 0, 'Unexpected library error', raising=exc.RuntimeError)