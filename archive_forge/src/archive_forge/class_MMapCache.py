from __future__ import annotations
import collections
import functools
import logging
import math
import os
import threading
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
class MMapCache(BaseCache):
    """memory-mapped sparse file cache

    Opens temporary file, which is filled blocks-wise when data is requested.
    Ensure there is enough disc space in the temporary location.

    This cache method might only work on posix
    """
    name = 'mmap'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, location: str | None=None, blocks: set[int] | None=None) -> None:
        super().__init__(blocksize, fetcher, size)
        self.blocks = set() if blocks is None else blocks
        self.location = location
        self.cache = self._makefile()

    def _makefile(self) -> mmap.mmap | bytearray:
        import mmap
        import tempfile
        if self.size == 0:
            return bytearray()
        if self.location is None or not os.path.exists(self.location):
            if self.location is None:
                fd = tempfile.TemporaryFile()
                self.blocks = set()
            else:
                fd = open(self.location, 'wb+')
            fd.seek(self.size - 1)
            fd.write(b'1')
            fd.flush()
        else:
            fd = open(self.location, 'r+b')
        return mmap.mmap(fd.fileno(), self.size)

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        logger.debug(f'MMap cache fetching {start}-{end}')
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b''
        start_block = start // self.blocksize
        end_block = end // self.blocksize
        need = [i for i in range(start_block, end_block + 1) if i not in self.blocks]
        while need:
            i = need.pop(0)
            sstart = i * self.blocksize
            send = min(sstart + self.blocksize, self.size)
            logger.debug(f'MMap get block #{i} ({sstart}-{send}')
            self.cache[sstart:send] = self.fetcher(sstart, send)
            self.blocks.add(i)
        return self.cache[start:end]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state['cache']
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.cache = self._makefile()