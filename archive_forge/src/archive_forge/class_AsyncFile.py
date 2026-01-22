from __future__ import annotations
import os
import pathlib
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from os import PathLike
from typing import (
from .. import to_thread
from ..abc import AsyncResource
class AsyncFile(AsyncResource, Generic[AnyStr]):
    """
    An asynchronous file object.

    This class wraps a standard file object and provides async friendly versions of the
    following blocking methods (where available on the original file object):

    * read
    * read1
    * readline
    * readlines
    * readinto
    * readinto1
    * write
    * writelines
    * truncate
    * seek
    * tell
    * flush

    All other methods are directly passed through.

    This class supports the asynchronous context manager protocol which closes the
    underlying file at the end of the context block.

    This class also supports asynchronous iteration::

        async with await open_file(...) as f:
            async for line in f:
                print(line)
    """

    def __init__(self, fp: IO[AnyStr]) -> None:
        self._fp: Any = fp

    def __getattr__(self, name: str) -> object:
        return getattr(self._fp, name)

    @property
    def wrapped(self) -> IO[AnyStr]:
        """The wrapped file object."""
        return self._fp

    async def __aiter__(self) -> AsyncIterator[AnyStr]:
        while True:
            line = await self.readline()
            if line:
                yield line
            else:
                break

    async def aclose(self) -> None:
        return await to_thread.run_sync(self._fp.close)

    async def read(self, size: int=-1) -> AnyStr:
        return await to_thread.run_sync(self._fp.read, size)

    async def read1(self: AsyncFile[bytes], size: int=-1) -> bytes:
        return await to_thread.run_sync(self._fp.read1, size)

    async def readline(self) -> AnyStr:
        return await to_thread.run_sync(self._fp.readline)

    async def readlines(self) -> list[AnyStr]:
        return await to_thread.run_sync(self._fp.readlines)

    async def readinto(self: AsyncFile[bytes], b: WriteableBuffer) -> bytes:
        return await to_thread.run_sync(self._fp.readinto, b)

    async def readinto1(self: AsyncFile[bytes], b: WriteableBuffer) -> bytes:
        return await to_thread.run_sync(self._fp.readinto1, b)

    @overload
    async def write(self: AsyncFile[bytes], b: ReadableBuffer) -> int:
        ...

    @overload
    async def write(self: AsyncFile[str], b: str) -> int:
        ...

    async def write(self, b: ReadableBuffer | str) -> int:
        return await to_thread.run_sync(self._fp.write, b)

    @overload
    async def writelines(self: AsyncFile[bytes], lines: Iterable[ReadableBuffer]) -> None:
        ...

    @overload
    async def writelines(self: AsyncFile[str], lines: Iterable[str]) -> None:
        ...

    async def writelines(self, lines: Iterable[ReadableBuffer] | Iterable[str]) -> None:
        return await to_thread.run_sync(self._fp.writelines, lines)

    async def truncate(self, size: int | None=None) -> int:
        return await to_thread.run_sync(self._fp.truncate, size)

    async def seek(self, offset: int, whence: int | None=os.SEEK_SET) -> int:
        return await to_thread.run_sync(self._fp.seek, offset, whence)

    async def tell(self) -> int:
        return await to_thread.run_sync(self._fp.tell)

    async def flush(self) -> None:
        return await to_thread.run_sync(self._fp.flush)