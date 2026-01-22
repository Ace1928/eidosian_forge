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
def _cached_custom_processor(self, dialect: Dialect, key: str, fn: Callable[[TypeEngine[_T]], _O]) -> _O:
    """return a dialect-specific processing object for
        custom purposes.

        The cx_Oracle dialect uses this at the moment.

        """
    try:
        return cast(_O, dialect._type_memos[self]['custom'][key])
    except KeyError:
        pass
    d = self._dialect_info(dialect)
    impl = d['impl']
    custom_dict = d.setdefault('custom', {})
    custom_dict[key] = result = fn(impl)
    return result