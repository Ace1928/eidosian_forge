from typing import ByteString, Optional, Tuple, cast
from nacl import exceptions as exc
from nacl._sodium import ffi, lib
from nacl.exceptions import ensure
class crypto_secretstream_xchacha20poly1305_state:
    """
    An object wrapping the crypto_secretstream_xchacha20poly1305 state.

    """
    __slots__ = ['statebuf', 'rawbuf', 'tagbuf']

    def __init__(self) -> None:
        """Initialize a clean state object."""
        self.statebuf: ByteString = ffi.new('unsigned char[]', crypto_secretstream_xchacha20poly1305_STATEBYTES)
        self.rawbuf: Optional[ByteString] = None
        self.tagbuf: Optional[ByteString] = None