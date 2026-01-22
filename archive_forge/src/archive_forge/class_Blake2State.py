from typing import NoReturn, TypeVar
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
class Blake2State:
    """
    Python-level wrapper for the crypto_generichash_blake2b state buffer
    """
    __slots__ = ['_statebuf', 'digest_size']

    def __init__(self, digest_size: int):
        self._statebuf = ffi.new('unsigned char[]', crypto_generichash_STATEBYTES)
        self.digest_size = digest_size

    def __reduce__(self) -> NoReturn:
        """
        Raise the same exception as hashlib's blake implementation
        on copy.copy()
        """
        raise TypeError("can't pickle {} objects".format(self.__class__.__name__))

    def copy(self: _Blake2State) -> _Blake2State:
        _st = self.__class__(self.digest_size)
        ffi.memmove(_st._statebuf, self._statebuf, crypto_generichash_STATEBYTES)
        return _st