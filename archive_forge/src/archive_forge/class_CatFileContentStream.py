from __future__ import annotations
import re
import contextlib
import io
import logging
import os
import signal
from subprocess import Popen, PIPE, DEVNULL
import subprocess
import threading
from textwrap import dedent
from git.compat import defenc, force_bytes, safe_decode
from git.exc import (
from git.util import (
from typing import (
from git.types import PathLike, Literal, TBD
class CatFileContentStream:
    """Object representing a sized read-only stream returning the contents of
        an object.

        This behaves like a stream, but counts the data read and simulates an empty
        stream once our sized content region is empty.

        If not all data are read to the end of the object's lifetime, we read the
        rest to ensure the underlying stream continues to work.
        """
    __slots__: Tuple[str, ...] = ('_stream', '_nbr', '_size')

    def __init__(self, size: int, stream: IO[bytes]) -> None:
        self._stream = stream
        self._size = size
        self._nbr = 0
        if size == 0:
            stream.read(1)

    def read(self, size: int=-1) -> bytes:
        bytes_left = self._size - self._nbr
        if bytes_left == 0:
            return b''
        if size > -1:
            size = min(bytes_left, size)
        else:
            size = bytes_left
        data = self._stream.read(size)
        self._nbr += len(data)
        if self._size - self._nbr == 0:
            self._stream.read(1)
        return data

    def readline(self, size: int=-1) -> bytes:
        if self._nbr == self._size:
            return b''
        bytes_left = self._size - self._nbr
        if size > -1:
            size = min(bytes_left, size)
        else:
            size = bytes_left
        data = self._stream.readline(size)
        self._nbr += len(data)
        if self._size - self._nbr == 0:
            self._stream.read(1)
        return data

    def readlines(self, size: int=-1) -> List[bytes]:
        if self._nbr == self._size:
            return []
        out = []
        nbr = 0
        while True:
            line = self.readline()
            if not line:
                break
            out.append(line)
            if size > -1:
                nbr += len(line)
                if nbr > size:
                    break
        return out

    def __iter__(self) -> 'Git.CatFileContentStream':
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line
    next = __next__

    def __del__(self) -> None:
        bytes_left = self._size - self._nbr
        if bytes_left:
            self._stream.read(bytes_left + 1)