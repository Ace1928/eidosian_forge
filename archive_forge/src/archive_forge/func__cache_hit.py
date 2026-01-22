import asyncio
import dataclasses
import sys
from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import (
def _cache_hit(self, key: Hashable) -> None:
    self.__hits += 1
    self.__cache.move_to_end(key)