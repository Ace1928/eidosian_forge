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
class InElementImpl(RoleImpl):
    __slots__ = ()

    def _implicit_coercions(self, element: Any, resolved: Any, argname: Optional[str]=None, **kw: Any) -> Any:
        if resolved._is_from_clause:
            if isinstance(resolved, selectable.Alias) and resolved.element._is_select_base:
                self._warn_for_implicit_coercion(resolved)
                return self._post_coercion(resolved.element, **kw)
            else:
                self._warn_for_implicit_coercion(resolved)
                return self._post_coercion(resolved.select(), **kw)
        else:
            self._raise_for_expected(element, argname, resolved)

    def _warn_for_implicit_coercion(self, elem):
        util.warn('Coercing %s object into a select() for use in IN(); please pass a select() construct explicitly' % elem.__class__.__name__)

    def _literal_coercion(self, element, expr, operator, **kw):
        if util.is_non_string_iterable(element):
            non_literal_expressions: Dict[Optional[operators.ColumnOperators], operators.ColumnOperators] = {}
            element = list(element)
            for o in element:
                if not _is_literal(o):
                    if not isinstance(o, operators.ColumnOperators):
                        self._raise_for_expected(element, **kw)
                    else:
                        non_literal_expressions[o] = o
                elif o is None:
                    non_literal_expressions[o] = elements.Null()
            if non_literal_expressions:
                return elements.ClauseList(*[non_literal_expressions[o] if o in non_literal_expressions else expr._bind_param(operator, o) for o in element])
            else:
                return expr._bind_param(operator, element, expanding=True)
        else:
            self._raise_for_expected(element, **kw)

    def _post_coercion(self, element, expr, operator, **kw):
        if element._is_select_base:
            return element.scalar_subquery()
        elif isinstance(element, elements.ClauseList):
            assert not len(element.clauses) == 0
            return element.self_group(against=operator)
        elif isinstance(element, elements.BindParameter):
            element = element._clone(maintain_key=True)
            element.expanding = True
            element.expand_op = operator
            return element
        elif isinstance(element, selectable.Values):
            return element.scalar_values()
        else:
            return element