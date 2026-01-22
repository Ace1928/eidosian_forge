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
def _populate_separate_keys(self, iter_: Iterable[Tuple[str, _NAMEDCOL]]) -> None:
    """populate from an iterator of (key, column)"""
    cols = list(iter_)
    replace_col = []
    for k, col in cols:
        if col.key != k:
            raise exc.ArgumentError('DedupeColumnCollection requires columns be under the same key as their .key')
        if col.name in self._index and col.key != col.name:
            replace_col.append(col)
        elif col.key in self._index:
            replace_col.append(col)
        else:
            self._index[k] = (k, col)
            self._collection.append((k, col, _ColumnMetrics(self, col)))
    self._colset.update((c._deannotate() for k, c, _ in self._collection))
    self._index.update(((idx, (k, c)) for idx, (k, c, _) in enumerate(self._collection)))
    for col in replace_col:
        self.replace(col)