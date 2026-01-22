from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
def _sodium_init() -> None:
    ensure(lib.sodium_init() != -1, 'Could not initialize sodium', raising=exc.RuntimeError)