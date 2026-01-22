from __future__ import annotations
import collections
from enum import Enum
import itertools
from itertools import zip_longest
import operator
import re
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
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import roles
from . import visitors
from .cache_key import HasCacheKey  # noqa
from .cache_key import MemoizedHasCacheKey  # noqa
from .traversals import HasCopyInternals  # noqa
from .visitors import ClauseVisitor
from .visitors import ExtendedInternalTraversal
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import util
from ..util import HasMemoized as HasMemoized
from ..util import hybridmethod
from ..util import typing as compat_typing
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeGuard
class ReadOnlyColumnCollection(util.ReadOnlyContainer, ColumnCollection[_COLKEY, _COL_co]):
    __slots__ = ('_parent',)

    def __init__(self, collection):
        object.__setattr__(self, '_parent', collection)
        object.__setattr__(self, '_colset', collection._colset)
        object.__setattr__(self, '_index', collection._index)
        object.__setattr__(self, '_collection', collection._collection)
        object.__setattr__(self, '_proxy_index', collection._proxy_index)

    def __getstate__(self):
        return {'_parent': self._parent}

    def __setstate__(self, state):
        parent = state['_parent']
        self.__init__(parent)

    def add(self, column: Any, key: Any=...) -> Any:
        self._readonly()

    def extend(self, elements: Any) -> NoReturn:
        self._readonly()

    def remove(self, item: Any) -> NoReturn:
        self._readonly()