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
def _cached_result_processor(self, dialect: Dialect, coltype: Any) -> Optional[_ResultProcessorType[_T]]:
    """Return a dialect-specific result processor for this type."""
    try:
        return dialect._type_memos[self]['result'][coltype]
    except KeyError:
        pass
    d = self._dialect_info(dialect)
    rp = d['impl'].result_processor(dialect, coltype)
    d['result'][coltype] = rp
    return rp