from types import coroutine
from collections.abc import Coroutine
from asyncio import get_running_loop
class AsyncBase:

    def __init__(self, file, loop, executor):
        self._file = file
        self._executor = executor
        self._ref_loop = loop

    @property
    def _loop(self):
        return self._ref_loop or get_running_loop()

    def __aiter__(self):
        """We are our own iterator."""
        return self

    def __repr__(self):
        return super().__repr__() + ' wrapping ' + repr(self._file)

    async def __anext__(self):
        """Simulate normal file iteration."""
        line = await self.readline()
        if line:
            return line
        else:
            raise StopAsyncIteration