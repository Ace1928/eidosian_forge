import asyncio
from typing import Awaitable, TypeVar, Optional, Callable
from google.api_core.exceptions import GoogleAPICallError, Unknown
from google.cloud.pubsublite.internal.wait_ignore_cancelled import wait_ignore_errors
class _TaskWithCleanup:

    def __init__(self, a: Awaitable):
        self._task = asyncio.ensure_future(a)

    async def __aenter__(self):
        return self._task

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._task.done():
            self._task.cancel()
            await wait_ignore_errors(self._task)