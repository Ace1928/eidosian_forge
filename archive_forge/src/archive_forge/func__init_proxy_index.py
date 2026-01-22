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
def _init_proxy_index(self):
    """populate the "proxy index", if empty.

        proxy index is added in 2.0 to provide more efficient operation
        for the corresponding_column() method.

        For reasons of both time to construct new .c collections as well as
        memory conservation for large numbers of large .c collections, the
        proxy_index is only filled if corresponding_column() is called. once
        filled it stays that way, and new _ColumnMetrics objects created after
        that point will populate it with new data. Note this case would be
        unusual, if not nonexistent, as it means a .c collection is being
        mutated after corresponding_column() were used, however it is tested in
        test/base/test_utils.py.

        """
    pi = self._proxy_index
    if pi:
        return
    for _, _, metrics in self._collection:
        eps = metrics.column._expanded_proxy_set
        for eps_col in eps:
            pi[eps_col].add(metrics)