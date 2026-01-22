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
def _validate_dialect_kwargs(self, kwargs: Dict[str, Any]) -> None:
    if not kwargs:
        return
    for k in kwargs:
        m = re.match('^(.+?)_(.+)$', k)
        if not m:
            raise TypeError("Additional arguments should be named <dialectname>_<argument>, got '%s'" % k)
        dialect_name, arg_name = m.group(1, 2)
        try:
            construct_arg_dictionary = self.dialect_options[dialect_name]
        except exc.NoSuchModuleError:
            util.warn("Can't validate argument %r; can't locate any SQLAlchemy dialect named %r" % (k, dialect_name))
            self.dialect_options[dialect_name] = d = _DialectArgDict()
            d._defaults.update({'*': None})
            d._non_defaults[arg_name] = kwargs[k]
        else:
            if '*' not in construct_arg_dictionary and arg_name not in construct_arg_dictionary:
                raise exc.ArgumentError('Argument %r is not accepted by dialect %r on behalf of %r' % (k, dialect_name, self.__class__))
            else:
                construct_arg_dictionary[arg_name] = kwargs[k]