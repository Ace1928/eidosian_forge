from __future__ import annotations
import operator
import threading
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from .base import NO_KEY
from .. import exc as sa_exc
from .. import util
from ..sql.base import NO_ARG
from ..util.compat import inspect_getfullargspec
from ..util.typing import Protocol
def _set_decorators() -> Dict[str, Callable[[_FN], _FN]]:
    """Tailored instrumentation wrappers for any set-like class."""

    def _tidy(fn):
        fn._sa_instrumented = True
        fn.__doc__ = getattr(set, fn.__name__).__doc__

    def add(fn):

        def add(self, value, _sa_initiator=None):
            if value not in self:
                value = __set(self, value, _sa_initiator, NO_KEY)
            else:
                __set_wo_mutation(self, value, _sa_initiator)
            fn(self, value)
        _tidy(add)
        return add

    def discard(fn):

        def discard(self, value, _sa_initiator=None):
            if value in self:
                __del(self, value, _sa_initiator, NO_KEY)
            fn(self, value)
        _tidy(discard)
        return discard

    def remove(fn):

        def remove(self, value, _sa_initiator=None):
            if value in self:
                __del(self, value, _sa_initiator, NO_KEY)
            fn(self, value)
        _tidy(remove)
        return remove

    def pop(fn):

        def pop(self):
            __before_pop(self)
            item = fn(self)
            __del(self, item, None, NO_KEY)
            return item
        _tidy(pop)
        return pop

    def clear(fn):

        def clear(self):
            for item in list(self):
                self.remove(item)
        _tidy(clear)
        return clear

    def update(fn):

        def update(self, value):
            for item in value:
                self.add(item)
        _tidy(update)
        return update

    def __ior__(fn):

        def __ior__(self, value):
            if not _set_binops_check_strict(self, value):
                return NotImplemented
            for item in value:
                self.add(item)
            return self
        _tidy(__ior__)
        return __ior__

    def difference_update(fn):

        def difference_update(self, value):
            for item in value:
                self.discard(item)
        _tidy(difference_update)
        return difference_update

    def __isub__(fn):

        def __isub__(self, value):
            if not _set_binops_check_strict(self, value):
                return NotImplemented
            for item in value:
                self.discard(item)
            return self
        _tidy(__isub__)
        return __isub__

    def intersection_update(fn):

        def intersection_update(self, other):
            want, have = (self.intersection(other), set(self))
            remove, add = (have - want, want - have)
            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)
        _tidy(intersection_update)
        return intersection_update

    def __iand__(fn):

        def __iand__(self, other):
            if not _set_binops_check_strict(self, other):
                return NotImplemented
            want, have = (self.intersection(other), set(self))
            remove, add = (have - want, want - have)
            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)
            return self
        _tidy(__iand__)
        return __iand__

    def symmetric_difference_update(fn):

        def symmetric_difference_update(self, other):
            want, have = (self.symmetric_difference(other), set(self))
            remove, add = (have - want, want - have)
            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)
        _tidy(symmetric_difference_update)
        return symmetric_difference_update

    def __ixor__(fn):

        def __ixor__(self, other):
            if not _set_binops_check_strict(self, other):
                return NotImplemented
            want, have = (self.symmetric_difference(other), set(self))
            remove, add = (have - want, want - have)
            for item in remove:
                self.remove(item)
            for item in add:
                self.add(item)
            return self
        _tidy(__ixor__)
        return __ixor__
    l = locals().copy()
    l.pop('_tidy')
    return l