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
class TestPmap:

    def test_ordering(self):
        """pmap results are in order, even if funcs finish out of order."""
        finished = []

        async def func(value):
            iterations = 10 - value
            for i in range(iterations):
                await duet.completed_future(i)
            finished.append(value)
            return value * 2
        results = duet.pmap(func, range(10), limit=10)
        assert results == [i * 2 for i in range(10)]
        assert finished == list(reversed(range(10)))

    @pytest.mark.parametrize('limit', [3, 10, None])
    def test_failure(self, limit):

        async def foo(i):
            if i == 7:
                raise ValueError('I do not like 7 :-(')
            return 7 * i
        with pytest.raises(ValueError):
            duet.pmap(foo, range(100), limit=limit)