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
def test_wrap_async_func(self):

    async def async_func(a, b):
        await duet.completed_future(None)
        return a + b
    assert duet.awaitable_func(async_func) is async_func
    assert duet.run(async_func, 1, 2) == 3