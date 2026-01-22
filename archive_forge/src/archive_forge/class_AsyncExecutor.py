import asyncio
import concurrent.futures
import logging
import queue
import sys
import threading
from typing import (
from wandb.errors.term import termerror
from wandb.filesync import upload_job
from wandb.sdk.lib.paths import LogicalPath
class AsyncExecutor:
    """Runs async file uploads in a background thread."""

    def __init__(self, pool: concurrent.futures.ThreadPoolExecutor, concurrency_limit: Optional[int]) -> None:
        self.loop = asyncio.new_event_loop()
        self.loop.set_default_executor(pool)
        self.loop_thread = threading.Thread(target=self.loop.run_forever, daemon=True, name='wandb-upload-async')
        self.concurrency_limiter = asyncio.Semaphore(value=concurrency_limit or 128, **{} if sys.version_info >= (3, 10) else {'loop': self.loop})

    def start(self) -> None:
        self.loop_thread.start()

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)

    def submit(self, coro: Awaitable[None]) -> None:

        async def run_with_limiter() -> None:
            async with self.concurrency_limiter:
                await coro
        asyncio.run_coroutine_threadsafe(run_with_limiter(), self.loop)