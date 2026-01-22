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
class _MetaOptions(type):
    """metaclass for the Options class.

    This metaclass is actually necessary despite the availability of the
    ``__init_subclass__()`` hook as this type also provides custom class-level
    behavior for the ``__add__()`` method.

    """
    _cache_attrs: Tuple[str, ...]

    def __add__(self, other):
        o1 = self()
        if set(other).difference(self._cache_attrs):
            raise TypeError('dictionary contains attributes not covered by Options class %s: %r' % (self, set(other).difference(self._cache_attrs)))
        o1.__dict__.update(other)
        return o1
    if TYPE_CHECKING:

        def __getattr__(self, key: str) -> Any:
            ...

        def __setattr__(self, key: str, value: Any) -> None:
            ...

        def __delattr__(self, key: str) -> None:
            ...