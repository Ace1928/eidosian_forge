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
def _dict_decorators() -> Dict[str, Callable[[_FN], _FN]]:
    """Tailored instrumentation wrappers for any dict-like mapping class."""

    def _tidy(fn):
        fn._sa_instrumented = True
        fn.__doc__ = getattr(dict, fn.__name__).__doc__

    def __setitem__(fn):

        def __setitem__(self, key, value, _sa_initiator=None):
            if key in self:
                __del(self, self[key], _sa_initiator, key)
            value = __set(self, value, _sa_initiator, key)
            fn(self, key, value)
        _tidy(__setitem__)
        return __setitem__

    def __delitem__(fn):

        def __delitem__(self, key, _sa_initiator=None):
            if key in self:
                __del(self, self[key], _sa_initiator, key)
            fn(self, key)
        _tidy(__delitem__)
        return __delitem__

    def clear(fn):

        def clear(self):
            for key in self:
                __del(self, self[key], None, key)
            fn(self)
        _tidy(clear)
        return clear

    def pop(fn):

        def pop(self, key, default=NO_ARG):
            __before_pop(self)
            _to_del = key in self
            if default is NO_ARG:
                item = fn(self, key)
            else:
                item = fn(self, key, default)
            if _to_del:
                __del(self, item, None, key)
            return item
        _tidy(pop)
        return pop

    def popitem(fn):

        def popitem(self):
            __before_pop(self)
            item = fn(self)
            __del(self, item[1], None, 1)
            return item
        _tidy(popitem)
        return popitem

    def setdefault(fn):

        def setdefault(self, key, default=None):
            if key not in self:
                self.__setitem__(key, default)
                return default
            else:
                value = self.__getitem__(key)
                if value is default:
                    __set_wo_mutation(self, value, None)
                return value
        _tidy(setdefault)
        return setdefault

    def update(fn):

        def update(self, __other=NO_ARG, **kw):
            if __other is not NO_ARG:
                if hasattr(__other, 'keys'):
                    for key in list(__other):
                        if key not in self or self[key] is not __other[key]:
                            self[key] = __other[key]
                        else:
                            __set_wo_mutation(self, __other[key], None)
                else:
                    for key, value in __other:
                        if key not in self or self[key] is not value:
                            self[key] = value
                        else:
                            __set_wo_mutation(self, value, None)
            for key in kw:
                if key not in self or self[key] is not kw[key]:
                    self[key] = kw[key]
                else:
                    __set_wo_mutation(self, kw[key], None)
        _tidy(update)
        return update
    l = locals().copy()
    l.pop('_tidy')
    return l