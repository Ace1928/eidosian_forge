import asyncio
import inspect
from collections import deque
from typing import (
class AsyncIteratorWrapper:

    def __init__(self, iterable: Iterable):
        self._iterator = iter(iterable)

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration

    def __aiter__(self):
        return self