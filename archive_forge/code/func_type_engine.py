from __future__ import annotations
from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
def type_engine(self, dialect: Dialect) -> TypeEngine[Any]:
    """Return a dialect-specific :class:`.TypeEngine` instance
        for this :class:`.TypeDecorator`.

        In most cases this returns a dialect-adapted form of
        the :class:`.TypeEngine` type represented by ``self.impl``.
        Makes usage of :meth:`dialect_impl`.
        Behavior can be customized here by overriding
        :meth:`load_dialect_impl`.

        """
    adapted = dialect.type_descriptor(self)
    if not isinstance(adapted, type(self)):
        return adapted
    else:
        return self.load_dialect_impl(dialect)