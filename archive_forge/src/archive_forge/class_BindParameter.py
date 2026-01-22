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
class BindParameter(roles.InElementRole, KeyedColumnElement[_T]):
    """Represent a "bound expression".

    :class:`.BindParameter` is invoked explicitly using the
    :func:`.bindparam` function, as in::

        from sqlalchemy import bindparam

        stmt = select(users_table).where(
            users_table.c.name == bindparam("username")
        )

    Detailed discussion of how :class:`.BindParameter` is used is
    at :func:`.bindparam`.

    .. seealso::

        :func:`.bindparam`

    """
    __visit_name__ = 'bindparam'
    _traverse_internals: _TraverseInternalsType = [('key', InternalTraversal.dp_anon_name), ('type', InternalTraversal.dp_type), ('callable', InternalTraversal.dp_plain_dict), ('value', InternalTraversal.dp_plain_obj), ('literal_execute', InternalTraversal.dp_boolean)]
    key: str
    type: TypeEngine[_T]
    value: Optional[_T]
    _is_crud = False
    _is_bind_parameter = True
    _key_is_anon = False
    inherit_cache = True

    def __init__(self, key: Optional[str], value: Any=_NoArg.NO_ARG, type_: Optional[_TypeEngineArgument[_T]]=None, unique: bool=False, required: Union[bool, Literal[_NoArg.NO_ARG]]=_NoArg.NO_ARG, quote: Optional[bool]=None, callable_: Optional[Callable[[], Any]]=None, expanding: bool=False, isoutparam: bool=False, literal_execute: bool=False, _compared_to_operator: Optional[OperatorType]=None, _compared_to_type: Optional[TypeEngine[Any]]=None, _is_crud: bool=False):
        if required is _NoArg.NO_ARG:
            required = value is _NoArg.NO_ARG and callable_ is None
        if value is _NoArg.NO_ARG:
            value = None
        if quote is not None:
            key = quoted_name.construct(key, quote)
        if unique:
            self.key = _anonymous_label.safe_construct(id(self), key if key is not None and (not isinstance(key, _anonymous_label)) else 'param', sanitize_key=True)
            self._key_is_anon = True
        elif key:
            self.key = key
        else:
            self.key = _anonymous_label.safe_construct(id(self), 'param')
            self._key_is_anon = True
        self._identifying_key = self.key
        self._orig_key = key or 'param'
        self.unique = unique
        self.value = value
        self.callable = callable_
        self.isoutparam = isoutparam
        self.required = required
        self.expanding = expanding
        self.expand_op = None
        self.literal_execute = literal_execute
        if _is_crud:
            self._is_crud = True
        if type_ is None:
            if expanding:
                if value:
                    check_value = value[0]
                else:
                    check_value = type_api._NO_VALUE_IN_LIST
            else:
                check_value = value
            if _compared_to_type is not None:
                self.type = _compared_to_type.coerce_compared_value(_compared_to_operator, check_value)
            else:
                self.type = type_api._resolve_value_to_type(check_value)
        elif isinstance(type_, type):
            self.type = type_()
        elif is_tuple_type(type_):
            if value:
                if expanding:
                    check_value = value[0]
                else:
                    check_value = value
                cast('BindParameter[typing_Tuple[Any, ...]]', self).type = type_._resolve_values_to_types(check_value)
            else:
                cast('BindParameter[typing_Tuple[Any, ...]]', self).type = type_
        else:
            self.type = type_

    def _with_value(self, value, maintain_key=False, required=NO_ARG):
        """Return a copy of this :class:`.BindParameter` with the given value
        set.
        """
        cloned = self._clone(maintain_key=maintain_key)
        cloned.value = value
        cloned.callable = None
        cloned.required = required if required is not NO_ARG else self.required
        if cloned.type is type_api.NULLTYPE:
            cloned.type = type_api._resolve_value_to_type(value)
        return cloned

    @property
    def effective_value(self) -> Optional[_T]:
        """Return the value of this bound parameter,
        taking into account if the ``callable`` parameter
        was set.

        The ``callable`` value will be evaluated
        and returned if present, else ``value``.

        """
        if self.callable:
            return self.callable()
        else:
            return self.value

    def render_literal_execute(self) -> BindParameter[_T]:
        """Produce a copy of this bound parameter that will enable the
        :paramref:`_sql.BindParameter.literal_execute` flag.

        The :paramref:`_sql.BindParameter.literal_execute` flag will
        have the effect of the parameter rendered in the compiled SQL
        string using ``[POSTCOMPILE]`` form, which is a special form that
        is converted to be a rendering of the literal value of the parameter
        at SQL execution time.    The rationale is to support caching
        of SQL statement strings that can embed per-statement literal values,
        such as LIMIT and OFFSET parameters, in the final SQL string that
        is passed to the DBAPI.   Dialects in particular may want to use
        this method within custom compilation schemes.

        .. versionadded:: 1.4.5

        .. seealso::

            :ref:`engine_thirdparty_caching`

        """
        c = ClauseElement._clone(self)
        c.literal_execute = True
        return c

    def _negate_in_binary(self, negated_op, original_op):
        if self.expand_op is original_op:
            bind = self._clone()
            bind.expand_op = negated_op
            return bind
        else:
            return self

    def _with_binary_element_type(self, type_):
        c = ClauseElement._clone(self)
        c.type = type_
        return c

    def _clone(self, maintain_key: bool=False, **kw: Any) -> Self:
        c = ClauseElement._clone(self, **kw)
        c._cloned_set.update(self._cloned_set)
        if not maintain_key and self.unique:
            c.key = _anonymous_label.safe_construct(id(c), c._orig_key or 'param', sanitize_key=True)
        return c

    def _gen_cache_key(self, anon_map, bindparams):
        _gen_cache_ok = self.__class__.__dict__.get('inherit_cache', False)
        if not _gen_cache_ok:
            if anon_map is not None:
                anon_map[NO_CACHE] = True
            return None
        id_, found = anon_map.get_anon(self)
        if found:
            return (id_, self.__class__)
        if bindparams is not None:
            bindparams.append(self)
        return (id_, self.__class__, self.type._static_cache_key, self.key % anon_map if self._key_is_anon else self.key, self.literal_execute)

    def _convert_to_unique(self):
        if not self.unique:
            self.unique = True
            self.key = _anonymous_label.safe_construct(id(self), self._orig_key or 'param', sanitize_key=True)

    def __getstate__(self):
        """execute a deferred value for serialization purposes."""
        d = self.__dict__.copy()
        v = self.value
        if self.callable:
            v = self.callable()
            d['callable'] = None
        d['value'] = v
        return d

    def __setstate__(self, state):
        if state.get('unique', False):
            state['key'] = _anonymous_label.safe_construct(id(self), state.get('_orig_key', 'param'), sanitize_key=True)
        self.__dict__.update(state)

    def __repr__(self):
        return '%s(%r, %r, type_=%r)' % (self.__class__.__name__, self.key, self.value, self.type)