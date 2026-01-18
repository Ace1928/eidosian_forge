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
def test_sync_on_classmethod(self):
    with pytest.raises(TypeError, match='duet.sync cannot be applied to classmethod'):

        class _Foo:

            @classmethod
            async def foo_async(cls, a: int) -> int:
                return a * 2
            foo = duet.sync(foo_async)