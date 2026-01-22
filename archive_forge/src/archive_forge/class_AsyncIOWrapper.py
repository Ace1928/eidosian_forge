from __future__ import annotations
import io
from functools import partial
from typing import (
import trio
from ._util import async_wraps
from .abc import AsyncResource
class AsyncIOWrapper(AsyncResource, Generic[FileT_co]):
    """A generic :class:`~io.IOBase` wrapper that implements the :term:`asynchronous
    file object` interface. Wrapped methods that could block are executed in
    :meth:`trio.to_thread.run_sync`.

    All properties and methods defined in :mod:`~io` are exposed by this
    wrapper, if they exist in the wrapped file object.
    """

    def __init__(self, file: FileT_co) -> None:
        self._wrapped = file

    @property
    def wrapped(self) -> FileT_co:
        """object: A reference to the wrapped file object"""
        return self._wrapped
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> object:
            if name in _FILE_SYNC_ATTRS:
                return getattr(self._wrapped, name)
            if name in _FILE_ASYNC_METHODS:
                meth = getattr(self._wrapped, name)

                @async_wraps(self.__class__, self._wrapped.__class__, name)
                async def wrapper(*args, **kwargs):
                    func = partial(meth, *args, **kwargs)
                    return await trio.to_thread.run_sync(func)
                setattr(self, name, wrapper)
                return wrapper
            raise AttributeError(name)

    def __dir__(self) -> Iterable[str]:
        attrs = set(super().__dir__())
        attrs.update((a for a in _FILE_SYNC_ATTRS if hasattr(self.wrapped, a)))
        attrs.update((a for a in _FILE_ASYNC_METHODS if hasattr(self.wrapped, a)))
        return attrs

    def __aiter__(self) -> AsyncIOWrapper[FileT_co]:
        return self

    async def __anext__(self: AsyncIOWrapper[_CanReadLine[AnyStr]]) -> AnyStr:
        line = await self.readline()
        if line:
            return line
        else:
            raise StopAsyncIteration

    async def detach(self: AsyncIOWrapper[_CanDetach[T]]) -> AsyncIOWrapper[T]:
        """Like :meth:`io.BufferedIOBase.detach`, but async.

        This also re-wraps the result in a new :term:`asynchronous file object`
        wrapper.

        """
        raw = await trio.to_thread.run_sync(self._wrapped.detach)
        return wrap_file(raw)

    async def aclose(self: AsyncIOWrapper[_CanClose]) -> None:
        """Like :meth:`io.IOBase.close`, but async.

        This is also shielded from cancellation; if a cancellation scope is
        cancelled, the wrapped file object will still be safely closed.

        """
        with trio.CancelScope(shield=True):
            await trio.to_thread.run_sync(self._wrapped.close)
        await trio.lowlevel.checkpoint_if_cancelled()
    if TYPE_CHECKING:

        @property
        def closed(self: AsyncIOWrapper[_HasClosed]) -> bool:
            ...

        @property
        def encoding(self: AsyncIOWrapper[_HasEncoding]) -> str:
            ...

        @property
        def errors(self: AsyncIOWrapper[_HasErrors]) -> str | None:
            ...

        @property
        def newlines(self: AsyncIOWrapper[_HasNewlines[T]]) -> T:
            ...

        @property
        def buffer(self: AsyncIOWrapper[_HasBuffer]) -> BinaryIO:
            ...

        @property
        def raw(self: AsyncIOWrapper[_HasRaw]) -> io.RawIOBase:
            ...

        @property
        def line_buffering(self: AsyncIOWrapper[_HasLineBuffering]) -> int:
            ...

        @property
        def closefd(self: AsyncIOWrapper[_HasCloseFD]) -> bool:
            ...

        @property
        def name(self: AsyncIOWrapper[_HasName]) -> str:
            ...

        @property
        def mode(self: AsyncIOWrapper[_HasMode]) -> str:
            ...

        def fileno(self: AsyncIOWrapper[_HasFileNo]) -> int:
            ...

        def isatty(self: AsyncIOWrapper[_HasIsATTY]) -> bool:
            ...

        def readable(self: AsyncIOWrapper[_HasReadable]) -> bool:
            ...

        def seekable(self: AsyncIOWrapper[_HasSeekable]) -> bool:
            ...

        def writable(self: AsyncIOWrapper[_HasWritable]) -> bool:
            ...

        def getvalue(self: AsyncIOWrapper[_CanGetValue[AnyStr]]) -> AnyStr:
            ...

        def getbuffer(self: AsyncIOWrapper[_CanGetBuffer]) -> memoryview:
            ...

        async def flush(self: AsyncIOWrapper[_CanFlush]) -> None:
            ...

        async def read(self: AsyncIOWrapper[_CanRead[AnyStr]], size: int | None=-1, /) -> AnyStr:
            ...

        async def read1(self: AsyncIOWrapper[_CanRead1], size: int | None=-1, /) -> bytes:
            ...

        async def readall(self: AsyncIOWrapper[_CanReadAll[AnyStr]]) -> AnyStr:
            ...

        async def readinto(self: AsyncIOWrapper[_CanReadInto], buf: Buffer, /) -> int | None:
            ...

        async def readline(self: AsyncIOWrapper[_CanReadLine[AnyStr]], size: int=-1, /) -> AnyStr:
            ...

        async def readlines(self: AsyncIOWrapper[_CanReadLines[AnyStr]]) -> list[AnyStr]:
            ...

        async def seek(self: AsyncIOWrapper[_CanSeek], target: int, whence: int=0, /) -> int:
            ...

        async def tell(self: AsyncIOWrapper[_CanTell]) -> int:
            ...

        async def truncate(self: AsyncIOWrapper[_CanTruncate], size: int | None=None, /) -> int:
            ...

        async def write(self: AsyncIOWrapper[_CanWrite[T]], data: T, /) -> int:
            ...

        async def writelines(self: AsyncIOWrapper[_CanWriteLines[T]], lines: Iterable[T], /) -> None:
            ...

        async def readinto1(self: AsyncIOWrapper[_CanReadInto1], buffer: Buffer, /) -> int:
            ...

        async def peek(self: AsyncIOWrapper[_CanPeek[AnyStr]], size: int=0, /) -> AnyStr:
            ...