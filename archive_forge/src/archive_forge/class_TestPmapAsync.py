import abc
import concurrent.futures
import contextlib
import inspect
import sys
import time
import traceback
from typing import List, Tuple
import pytest
import duet
import duet.impl as impl
class TestPmapAsync:

    @duet.sync
    async def test_ordering(self):
        """pmap_async results in order, even if funcs finish out of order."""
        finished = []

        async def func(value):
            iterations = 10 - value
            for i in range(iterations):
                await duet.completed_future(i)
            finished.append(value)
            return value * 2
        results = await duet.pmap_async(func, range(10), limit=10)
        assert results == [i * 2 for i in range(10)]
        assert finished == list(reversed(range(10)))

    @duet.sync
    async def test_laziness(self):
        live = set()

        async def func(i):
            num_live = len(live)
            live.add(i)
            await duet.completed_future(i)
            live.remove(i)
            return num_live
        num_lives = await duet.pmap_async(func, range(100), limit=10)
        assert all((num_live <= 10 for num_live in num_lives))