import asyncio
import dataclasses
import sys
from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import (
@final
class _CacheParameters(TypedDict):
    typed: bool
    maxsize: Optional[int]
    tasks: int
    closed: bool