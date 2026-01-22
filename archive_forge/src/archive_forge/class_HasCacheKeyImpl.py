from __future__ import annotations
import collections.abc as collections_abc
import numbers
import re
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import operators
from . import roles
from . import visitors
from ._typing import is_from_clause
from .base import ExecutableOption
from .base import Options
from .cache_key import HasCacheKey
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util.typing import Literal
class HasCacheKeyImpl(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(self, element: Any, resolved: Any, argname: Optional[str]=None, **kw: Any) -> Any:
        if isinstance(element, HasCacheKey):
            return element
        else:
            self._raise_for_expected(element, argname, resolved)

    def _literal_coercion(self, element, **kw):
        return element