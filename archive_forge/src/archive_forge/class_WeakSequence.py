from __future__ import annotations
import operator
import threading
import types
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import ValuesView
import weakref
from ._has_cy import HAS_CYEXTENSION
from .typing import is_non_string_iterable
from .typing import Literal
from .typing import Protocol
class WeakSequence(Sequence[_T]):

    def __init__(self, __elements: Sequence[_T]=()):

        def _remove(item, selfref=weakref.ref(self)):
            self = selfref()
            if self is not None:
                self._storage.remove(item)
        self._remove = _remove
        self._storage = [weakref.ref(element, _remove) for element in __elements]

    def append(self, item):
        self._storage.append(weakref.ref(item, self._remove))

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return (obj for obj in (ref() for ref in self._storage) if obj is not None)

    def __getitem__(self, index):
        try:
            obj = self._storage[index]
        except KeyError:
            raise IndexError('Index %s out of range' % index)
        else:
            return obj()