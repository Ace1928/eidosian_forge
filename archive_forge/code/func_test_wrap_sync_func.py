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
def test_wrap_sync_func(self):

    def sync_func(a, b):
        return a + b
    wrapped = duet.awaitable_func(sync_func)
    assert inspect.iscoroutinefunction(wrapped)
    assert duet.awaitable_func(wrapped) is wrapped
    assert duet.run(wrapped, 1, 2) == 3