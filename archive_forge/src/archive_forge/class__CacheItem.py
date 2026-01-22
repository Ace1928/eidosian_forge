import asyncio
import dataclasses
import sys
from asyncio.coroutines import _is_coroutine  # type: ignore[attr-defined]
from functools import _CacheInfo, _make_key, partial, partialmethod
from typing import (
@final
@dataclasses.dataclass
class _CacheItem(Generic[_R]):
    fut: 'asyncio.Future[_R]'
    later_call: Optional[asyncio.Handle]

    def cancel(self) -> None:
        if self.later_call is not None:
            self.later_call.cancel()
            self.later_call = None