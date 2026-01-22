from __future__ import annotations
from collections.abc import Callable, Mapping
from io import SEEK_SET, UnsupportedOperation
from os import PathLike
from pathlib import Path
from typing import Any, BinaryIO, cast
from .. import (
from ..abc import ByteReceiveStream, ByteSendStream
class FileReadStream(_BaseFileStream, ByteReceiveStream):
    """
    A byte stream that reads from a file in the file system.

    :param file: a file that has been opened for reading in binary mode

    .. versionadded:: 3.0
    """

    @classmethod
    async def from_path(cls, path: str | PathLike[str]) -> FileReadStream:
        """
        Create a file read stream by opening the given file.

        :param path: path of the file to read from

        """
        file = await to_thread.run_sync(Path(path).open, 'rb')
        return cls(cast(BinaryIO, file))

    async def receive(self, max_bytes: int=65536) -> bytes:
        try:
            data = await to_thread.run_sync(self._file.read, max_bytes)
        except ValueError:
            raise ClosedResourceError from None
        except OSError as exc:
            raise BrokenResourceError from exc
        if data:
            return data
        else:
            raise EndOfStream

    async def seek(self, position: int, whence: int=SEEK_SET) -> int:
        """
        Seek the file to the given position.

        .. seealso:: :meth:`io.IOBase.seek`

        .. note:: Not all file descriptors are seekable.

        :param position: position to seek the file to
        :param whence: controls how ``position`` is interpreted
        :return: the new absolute position
        :raises OSError: if the file is not seekable

        """
        return await to_thread.run_sync(self._file.seek, position, whence)

    async def tell(self) -> int:
        """
        Return the current stream position.

        .. note:: Not all file descriptors are seekable.

        :return: the current absolute position
        :raises OSError: if the file is not seekable

        """
        return await to_thread.run_sync(self._file.tell)