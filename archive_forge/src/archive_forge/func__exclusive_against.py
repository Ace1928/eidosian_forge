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
def _exclusive_against(*names: str, **kw: Any) -> Callable[[_Fn], _Fn]:
    msgs = kw.pop('msgs', {})
    defaults = kw.pop('defaults', {})
    getters = [(name, operator.attrgetter(name), defaults.get(name, None)) for name in names]

    @util.decorator
    def check(fn, *args, **kw):
        self = args[0]
        args = args[1:]
        for name, getter, default_ in getters:
            if getter(self) is not default_:
                msg = msgs.get(name, 'Method %s() has already been invoked on this %s construct' % (fn.__name__, self.__class__))
                raise exc.InvalidRequestError(msg)
        return fn(self, *args, **kw)
    return check