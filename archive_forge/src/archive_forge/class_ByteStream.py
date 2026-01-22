from typing import (
from urllib.parse import urlparse
class ByteStream:
    """
    A container for non-streaming content, and that supports both sync and async
    stream iteration.
    """

    def __init__(self, content: bytes) -> None:
        self._content = content

    def __iter__(self) -> Iterator[bytes]:
        yield self._content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._content

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{len(self._content)} bytes]>'