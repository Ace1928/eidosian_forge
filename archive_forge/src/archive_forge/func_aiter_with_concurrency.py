import asyncio
import inspect
from collections import deque
from typing import (
def aiter_with_concurrency(n: Optional[int], generator: AsyncIterator[Coroutine[None, None, T]]) -> AsyncGenerator[T, None]:
    """Process async generator with max parallelism.

    Args:
        n: The number of tasks to run concurrently.
        generator: The async generator to process.

    Yields:
        The processed items yielded by the async generator.
    """
    if n == 0:

        async def consume():
            async for item in generator:
                yield (await item)
        return consume()
    semaphore = asyncio.Semaphore(n) if n is not None else NoLock()

    async def process_item(item):
        async with semaphore:
            return await item

    async def process_generator():
        tasks = []
        async for item in generator:
            task = asyncio.create_task(process_item(item))
            tasks.append(task)
            if n is not None and len(tasks) >= n:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                for task in done:
                    yield task.result()
        for task in asyncio.as_completed(tasks):
            yield (await task)
    return process_generator()