from __future__ import annotations
from collections.abc import Callable
from contextlib import contextmanager
from typing import ClassVar
def unpack_callbacks(cbs):
    """Take an iterable of callbacks, return a list of each callback."""
    if cbs:
        return [[i for i in f if i] for f in zip(*cbs)]
    else:
        return [(), (), (), (), ()]