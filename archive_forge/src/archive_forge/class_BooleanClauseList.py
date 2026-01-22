from __future__ import annotations
from decimal import Decimal
from enum import IntEnum
import itertools
import operator
import re
import typing
from typing import AbstractSet
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
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple as typing_Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from ._typing import has_schema_attr
from ._typing import is_named_from_clause
from ._typing import is_quoted_name
from ._typing import is_tuple_type
from .annotation import Annotated
from .annotation import SupportsWrappingAnnotations
from .base import _clone
from .base import _expand_cloned
from .base import _generative
from .base import _NoArg
from .base import Executable
from .base import Generative
from .base import HasMemoized
from .base import Immutable
from .base import NO_ARG
from .base import SingletonConstant
from .cache_key import MemoizedHasCacheKey
from .cache_key import NO_CACHE
from .coercions import _document_text_coercion  # noqa
from .operators import ColumnOperators
from .traversals import HasCopyInternals
from .visitors import cloned_traverse
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .visitors import traverse
from .visitors import Visitable
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util import TypingOnly
from ..util.typing import Literal
from ..util.typing import Self
class BooleanClauseList(ExpressionClauseList[bool]):
    __visit_name__ = 'expression_clauselist'
    inherit_cache = True

    def __init__(self, *arg, **kw):
        raise NotImplementedError('BooleanClauseList has a private constructor')

    @classmethod
    def _process_clauses_for_boolean(cls, operator: OperatorType, continue_on: Any, skip_on: Any, clauses: Iterable[ColumnElement[Any]]) -> typing_Tuple[int, List[ColumnElement[Any]]]:
        has_continue_on = None
        convert_clauses = []
        against = operators._asbool
        lcc = 0
        for clause in clauses:
            if clause is continue_on:
                has_continue_on = clause
            elif clause is skip_on:
                convert_clauses = [clause]
                lcc = 1
                break
            else:
                if not lcc:
                    lcc = 1
                else:
                    against = operator
                    lcc = 2
                convert_clauses.append(clause)
        if not convert_clauses and has_continue_on is not None:
            convert_clauses = [has_continue_on]
            lcc = 1
        return (lcc, [c.self_group(against=against) for c in convert_clauses])

    @classmethod
    def _construct(cls, operator: OperatorType, continue_on: Any, skip_on: Any, initial_clause: Any=_NoArg.NO_ARG, *clauses: Any, **kw: Any) -> ColumnElement[Any]:
        if initial_clause is _NoArg.NO_ARG:
            name = operator.__name__
            util.warn_deprecated(f"Invoking {name}() without arguments is deprecated, and will be disallowed in a future release.   For an empty {name}() construct, use '{name}({('true()' if continue_on is True_._singleton else 'false()')}, *args)' or '{name}({('True' if continue_on is True_._singleton else 'False')}, *args)'.", version='1.4')
            return cls._construct_raw(operator)
        lcc, convert_clauses = cls._process_clauses_for_boolean(operator, continue_on, skip_on, [coercions.expect(roles.WhereHavingRole, clause) for clause in util.coerce_generator_arg((initial_clause,) + clauses)])
        if lcc > 1:
            flattened_clauses = itertools.chain.from_iterable(((c for c in to_flat._flattened_operator_clauses) if getattr(to_flat, 'operator', None) is operator else (to_flat,) for to_flat in convert_clauses))
            return cls._construct_raw(operator, flattened_clauses)
        else:
            assert lcc
            return convert_clauses[0]

    @classmethod
    def _construct_for_whereclause(cls, clauses: Iterable[ColumnElement[Any]]) -> Optional[ColumnElement[bool]]:
        operator, continue_on, skip_on = (operators.and_, True_._singleton, False_._singleton)
        lcc, convert_clauses = cls._process_clauses_for_boolean(operator, continue_on, skip_on, clauses)
        if lcc > 1:
            return cls._construct_raw(operator, convert_clauses)
        elif lcc == 1:
            return convert_clauses[0]
        else:
            return None

    @classmethod
    def _construct_raw(cls, operator: OperatorType, clauses: Optional[Sequence[ColumnElement[Any]]]=None) -> BooleanClauseList:
        self = cls.__new__(cls)
        self.clauses = tuple(clauses) if clauses else ()
        self.group = True
        self.operator = operator
        self.type = type_api.BOOLEANTYPE
        self._is_implicitly_boolean = True
        return self

    @classmethod
    def and_(cls, initial_clause: Union[Literal[True], _ColumnExpressionArgument[bool], _NoArg]=_NoArg.NO_ARG, *clauses: _ColumnExpressionArgument[bool]) -> ColumnElement[bool]:
        """Produce a conjunction of expressions joined by ``AND``.

        See :func:`_sql.and_` for full documentation.
        """
        return cls._construct(operators.and_, True_._singleton, False_._singleton, initial_clause, *clauses)

    @classmethod
    def or_(cls, initial_clause: Union[Literal[False], _ColumnExpressionArgument[bool], _NoArg]=_NoArg.NO_ARG, *clauses: _ColumnExpressionArgument[bool]) -> ColumnElement[bool]:
        """Produce a conjunction of expressions joined by ``OR``.

        See :func:`_sql.or_` for full documentation.
        """
        return cls._construct(operators.or_, False_._singleton, True_._singleton, initial_clause, *clauses)

    @property
    def _select_iterable(self) -> _SelectIterable:
        return (self,)

    def self_group(self, against=None):
        if not self.clauses:
            return self
        else:
            return super().self_group(against=against)