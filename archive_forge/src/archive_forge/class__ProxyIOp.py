from __future__ import annotations
import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter
from .wsgi import ClosingIterator
class _ProxyIOp(_ProxyLookup):
    """Look up an augmented assignment method on a proxied object. The
    method is wrapped to return the proxy instead of the object.
    """
    __slots__ = ()

    def __init__(self, f: t.Callable | None=None, fallback: t.Callable | None=None) -> None:
        super().__init__(f, fallback)

        def bind_f(instance: LocalProxy, obj: t.Any) -> t.Callable:

            def i_op(self: t.Any, other: t.Any) -> LocalProxy:
                f(self, other)
                return instance
            return i_op.__get__(obj, type(obj))
        self.bind_f = bind_f