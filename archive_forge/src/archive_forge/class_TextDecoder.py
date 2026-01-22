from __future__ import annotations
import codecs
import io
import typing
import zlib
from ._compat import brotli
from ._exceptions import DecodingError
class TextDecoder:
    """
    Handles incrementally decoding bytes into text
    """

    def __init__(self, encoding: str='utf-8') -> None:
        self.decoder = codecs.getincrementaldecoder(encoding)(errors='replace')

    def decode(self, data: bytes) -> str:
        return self.decoder.decode(data)

    def flush(self) -> str:
        return self.decoder.decode(b'', True)