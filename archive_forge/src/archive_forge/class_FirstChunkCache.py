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
class FirstChunkCache(BaseCache):
    """Caches the first block of a file only

    This may be useful for file types where the metadata is stored in the header,
    but is randomly accessed.
    """
    name = 'first'

    def __init__(self, blocksize: int, fetcher: Fetcher, size: int) -> None:
        super().__init__(blocksize, fetcher, size)
        self.cache: bytes | None = None

    def _fetch(self, start: int | None, end: int | None) -> bytes:
        start = start or 0
        end = end or self.size
        if start < self.blocksize:
            if self.cache is None:
                if end > self.blocksize:
                    data = self.fetcher(0, end)
                    self.cache = data[:self.blocksize]
                    return data[start:]
                self.cache = self.fetcher(0, self.blocksize)
            part = self.cache[start:end]
            if end > self.blocksize:
                part += self.fetcher(self.blocksize, end)
            return part
        else:
            return self.fetcher(start, end)