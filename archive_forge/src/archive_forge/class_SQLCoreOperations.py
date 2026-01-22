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
class SQLCoreOperations(Generic[_T_co], ColumnOperators, TypingOnly):
    __slots__ = ()
    if typing.TYPE_CHECKING:

        @util.non_memoized_property
        def _propagate_attrs(self) -> _PropagateAttrsType:
            ...

        def operate(self, op: OperatorType, *other: Any, **kwargs: Any) -> ColumnElement[Any]:
            ...

        def reverse_operate(self, op: OperatorType, other: Any, **kwargs: Any) -> ColumnElement[Any]:
            ...

        @overload
        def op(self, opstring: str, precedence: int=..., is_comparison: bool=..., *, return_type: _TypeEngineArgument[_OPT], python_impl: Optional[Callable[..., Any]]=None) -> Callable[[Any], BinaryExpression[_OPT]]:
            ...

        @overload
        def op(self, opstring: str, precedence: int=..., is_comparison: bool=..., return_type: Optional[_TypeEngineArgument[Any]]=..., python_impl: Optional[Callable[..., Any]]=...) -> Callable[[Any], BinaryExpression[Any]]:
            ...

        def op(self, opstring: str, precedence: int=0, is_comparison: bool=False, return_type: Optional[_TypeEngineArgument[Any]]=None, python_impl: Optional[Callable[..., Any]]=None) -> Callable[[Any], BinaryExpression[Any]]:
            ...

        def bool_op(self, opstring: str, precedence: int=0, python_impl: Optional[Callable[..., Any]]=None) -> Callable[[Any], BinaryExpression[bool]]:
            ...

        def __and__(self, other: Any) -> BooleanClauseList:
            ...

        def __or__(self, other: Any) -> BooleanClauseList:
            ...

        def __invert__(self) -> ColumnElement[_T_co]:
            ...

        def __lt__(self, other: Any) -> ColumnElement[bool]:
            ...

        def __le__(self, other: Any) -> ColumnElement[bool]:
            ...

        def __hash__(self) -> int:
            ...

        def __eq__(self, other: Any) -> ColumnElement[bool]:
            ...

        def __ne__(self, other: Any) -> ColumnElement[bool]:
            ...

        def is_distinct_from(self, other: Any) -> ColumnElement[bool]:
            ...

        def is_not_distinct_from(self, other: Any) -> ColumnElement[bool]:
            ...

        def __gt__(self, other: Any) -> ColumnElement[bool]:
            ...

        def __ge__(self, other: Any) -> ColumnElement[bool]:
            ...

        def __neg__(self) -> UnaryExpression[_T_co]:
            ...

        def __contains__(self, other: Any) -> ColumnElement[bool]:
            ...

        def __getitem__(self, index: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __lshift__(self: _SQO[int], other: Any) -> ColumnElement[int]:
            ...

        @overload
        def __lshift__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __lshift__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __rshift__(self: _SQO[int], other: Any) -> ColumnElement[int]:
            ...

        @overload
        def __rshift__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __rshift__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def concat(self: _SQO[str], other: Any) -> ColumnElement[str]:
            ...

        @overload
        def concat(self, other: Any) -> ColumnElement[Any]:
            ...

        def concat(self, other: Any) -> ColumnElement[Any]:
            ...

        def like(self, other: Any, escape: Optional[str]=None) -> BinaryExpression[bool]:
            ...

        def ilike(self, other: Any, escape: Optional[str]=None) -> BinaryExpression[bool]:
            ...

        def bitwise_xor(self, other: Any) -> BinaryExpression[Any]:
            ...

        def bitwise_or(self, other: Any) -> BinaryExpression[Any]:
            ...

        def bitwise_and(self, other: Any) -> BinaryExpression[Any]:
            ...

        def bitwise_not(self) -> UnaryExpression[_T_co]:
            ...

        def bitwise_lshift(self, other: Any) -> BinaryExpression[Any]:
            ...

        def bitwise_rshift(self, other: Any) -> BinaryExpression[Any]:
            ...

        def in_(self, other: Union[Iterable[Any], BindParameter[Any], roles.InElementRole]) -> BinaryExpression[bool]:
            ...

        def not_in(self, other: Union[Iterable[Any], BindParameter[Any], roles.InElementRole]) -> BinaryExpression[bool]:
            ...

        def notin_(self, other: Union[Iterable[Any], BindParameter[Any], roles.InElementRole]) -> BinaryExpression[bool]:
            ...

        def not_like(self, other: Any, escape: Optional[str]=None) -> BinaryExpression[bool]:
            ...

        def notlike(self, other: Any, escape: Optional[str]=None) -> BinaryExpression[bool]:
            ...

        def not_ilike(self, other: Any, escape: Optional[str]=None) -> BinaryExpression[bool]:
            ...

        def notilike(self, other: Any, escape: Optional[str]=None) -> BinaryExpression[bool]:
            ...

        def is_(self, other: Any) -> BinaryExpression[bool]:
            ...

        def is_not(self, other: Any) -> BinaryExpression[bool]:
            ...

        def isnot(self, other: Any) -> BinaryExpression[bool]:
            ...

        def startswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnElement[bool]:
            ...

        def istartswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnElement[bool]:
            ...

        def endswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnElement[bool]:
            ...

        def iendswith(self, other: Any, escape: Optional[str]=None, autoescape: bool=False) -> ColumnElement[bool]:
            ...

        def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
            ...

        def icontains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
            ...

        def match(self, other: Any, **kwargs: Any) -> ColumnElement[bool]:
            ...

        def regexp_match(self, pattern: Any, flags: Optional[str]=None) -> ColumnElement[bool]:
            ...

        def regexp_replace(self, pattern: Any, replacement: Any, flags: Optional[str]=None) -> ColumnElement[str]:
            ...

        def desc(self) -> UnaryExpression[_T_co]:
            ...

        def asc(self) -> UnaryExpression[_T_co]:
            ...

        def nulls_first(self) -> UnaryExpression[_T_co]:
            ...

        def nullsfirst(self) -> UnaryExpression[_T_co]:
            ...

        def nulls_last(self) -> UnaryExpression[_T_co]:
            ...

        def nullslast(self) -> UnaryExpression[_T_co]:
            ...

        def collate(self, collation: str) -> CollationClause:
            ...

        def between(self, cleft: Any, cright: Any, symmetric: bool=False) -> BinaryExpression[bool]:
            ...

        def distinct(self: _SQO[_T_co]) -> UnaryExpression[_T_co]:
            ...

        def any_(self) -> CollectionAggregate[Any]:
            ...

        def all_(self) -> CollectionAggregate[Any]:
            ...

        @overload
        def __add__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __add__(self: _SQO[str], other: Any) -> ColumnElement[str]:
            ...

        def __add__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __radd__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __radd__(self: _SQO[str], other: Any) -> ColumnElement[str]:
            ...

        def __radd__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __sub__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __sub__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __sub__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __rsub__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __rsub__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __rsub__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __mul__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __mul__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __mul__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __rmul__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __rmul__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __rmul__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __mod__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __mod__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __mod__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __rmod__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __rmod__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __rmod__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __truediv__(self: _SQO[int], other: Any) -> ColumnElement[_NUMERIC]:
            ...

        @overload
        def __truediv__(self: _SQO[_NT], other: Any) -> ColumnElement[_NT]:
            ...

        @overload
        def __truediv__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __truediv__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __rtruediv__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NUMERIC]:
            ...

        @overload
        def __rtruediv__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __rtruediv__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __floordiv__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __floordiv__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __floordiv__(self, other: Any) -> ColumnElement[Any]:
            ...

        @overload
        def __rfloordiv__(self: _SQO[_NMT], other: Any) -> ColumnElement[_NMT]:
            ...

        @overload
        def __rfloordiv__(self, other: Any) -> ColumnElement[Any]:
            ...

        def __rfloordiv__(self, other: Any) -> ColumnElement[Any]:
            ...