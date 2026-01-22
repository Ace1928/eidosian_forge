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
class BytesCache(BaseCache):
    """Cache which holds data in a in-memory bytes object

    Implements read-ahead by the block size, for semi-random reads progressing
    through the file.

    Parameters
    ----------
    trim: bool
        As we read more data, whether to discard the start of the buffer when
        we are more than a blocksize ahead of it.
    """
    name: ClassVar[str] = 'bytes'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int, trim: bool=True) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache = b''
        self.start: int | None = None
        self.end: int | None = None
        self.trim = trim

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b''
        if self.start is not None and start >= self.start and (self.end is not None) and (end < self.end):
            offset = start - self.start
            return self.cache[offset:offset + end - start]
        if self.blocksize:
            bend = min(self.size, end + self.blocksize)
        else:
            bend = end
        if bend == start or start > self.size:
            return b''
        if (self.start is None or start < self.start) and (self.end is None or end > self.end):
            self.cache = self.fetcher(start, bend)
            self.start = start
        else:
            assert self.start is not None
            assert self.end is not None
            if start < self.start:
                if self.end is None or self.end - end > self.blocksize:
                    self.cache = self.fetcher(start, bend)
                    self.start = start
                else:
                    new = self.fetcher(start, self.start)
                    self.start = start
                    self.cache = new + self.cache
            elif self.end is not None and bend > self.end:
                if self.end > self.size:
                    pass
                elif end - self.end > self.blocksize:
                    self.cache = self.fetcher(start, bend)
                    self.start = start
                else:
                    new = self.fetcher(self.end, bend)
                    self.cache = self.cache + new
        self.end = self.start + len(self.cache)
        offset = start - self.start
        out = self.cache[offset:offset + end - start]
        if self.trim:
            num = (self.end - self.start) // (self.blocksize + 1)
            if num > 1:
                self.start += self.blocksize * num
                self.cache = self.cache[self.blocksize * num:]
        return out

    def __len__(self) -> int:
        return len(self.cache)