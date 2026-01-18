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
def test_sync_on_overridden_method(self):

    class Foo:

        async def foo_async(self, a: int) -> int:
            return a * 2
        foo = duet.sync(foo_async)

    class Bar(Foo):

        async def foo_async(self, a: int) -> int:
            return a * 3
    assert Foo().foo(5) == 10
    assert Bar().foo(5) == 15