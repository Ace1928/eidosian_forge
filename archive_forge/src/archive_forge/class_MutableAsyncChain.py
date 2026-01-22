import collections.abc
import gc
import inspect
import re
import sys
import weakref
from functools import partial, wraps
from itertools import chain
from typing import (
from scrapy.utils.asyncgen import as_async_generator
class MutableAsyncChain(AsyncIterable):
    """
    Similar to MutableChain but for async iterables
    """

    def __init__(self, *args: Union[Iterable, AsyncIterable]):
        self.data = _async_chain(*args)

    def extend(self, *iterables: Union[Iterable, AsyncIterable]) -> None:
        self.data = _async_chain(self.data, _async_chain(*iterables))

    def __aiter__(self) -> AsyncIterator:
        return self

    async def __anext__(self) -> Any:
        return await self.data.__anext__()