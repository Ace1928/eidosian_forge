import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _enter(self):
    while self._state in (-1, 1):
        self._cond.wait()
    if self._state < 0:
        raise BrokenBarrierError
    assert self._state == 0