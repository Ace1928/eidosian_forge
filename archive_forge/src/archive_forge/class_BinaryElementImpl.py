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
class BinaryElementImpl(ExpressionElementImpl, RoleImpl):
    __slots__ = ()

    def _literal_coercion(self, element, expr, operator, bindparam_type=None, argname=None, **kw):
        try:
            return expr._bind_param(operator, element, type_=bindparam_type)
        except exc.ArgumentError as err:
            self._raise_for_expected(element, err=err)

    def _post_coercion(self, resolved, expr, bindparam_type=None, **kw):
        if resolved.type._isnull and (not expr.type._isnull):
            resolved = resolved._with_binary_element_type(bindparam_type if bindparam_type is not None else expr.type)
        return resolved