from collections import deque
from enum import Enum, IntEnum, IntFlag
import struct
from typing import Optional
class BufferPipe:
    """A place to store received data until we can parse a complete message

    The main difference from io.BytesIO is that read & write operate at
    opposite ends, like a pipe.
    """

    def __init__(self):
        self.chunks = deque()
        self.bytes_buffered = 0

    def write(self, b: bytes):
        self.chunks.append(b)
        self.bytes_buffered += len(b)

    def _peek_iter(self, nbytes: int):
        assert nbytes <= self.bytes_buffered
        for chunk in self.chunks:
            chunk = chunk[:nbytes]
            nbytes -= len(chunk)
            yield chunk
            if nbytes <= 0:
                break

    def peek(self, nbytes: int) -> bytes:
        """Get exactly nbytes bytes from the front without removing them"""
        return b''.join(self._peek_iter(nbytes))

    def _read_iter(self, nbytes: int):
        assert nbytes <= self.bytes_buffered
        while True:
            chunk = self.chunks.popleft()
            self.bytes_buffered -= len(chunk)
            if nbytes <= len(chunk):
                break
            nbytes -= len(chunk)
            yield chunk
        chunk, rem = (chunk[:nbytes], chunk[nbytes:])
        if rem:
            self.chunks.appendleft(rem)
            self.bytes_buffered += len(rem)
        yield chunk

    def read(self, nbytes: int) -> bytes:
        """Take & return exactly nbytes bytes from the front"""
        return b''.join(self._read_iter(nbytes))