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
class TestPstarmapAsync:

    @duet.sync
    async def test_ordering(self):
        """pstarmap_async results in order, even if funcs finish out of order."""
        finished = []

        async def func(a, b):
            value = 5 * a + b
            iterations = 10 - value
            for i in range(iterations):
                await duet.completed_future(i)
            finished.append(value)
            return value * 2
        args_iter = ((a, b) for a in range(2) for b in range(5))
        results = await duet.pstarmap_async(func, args_iter, limit=10)
        assert results == [i * 2 for i in range(10)]
        assert finished == list(reversed(range(10)))