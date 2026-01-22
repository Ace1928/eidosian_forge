from __future__ import annotations
import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import Executable
from .base import NO_ARG
from .elements import ClauseElement
from .elements import quoted_name
from .schema import Column
from .sqltypes import TupleType
from .type_api import TypeEngine
from .visitors import prefix_anon_map
from .visitors import Visitable
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import TypedDict
def _label_select_column(self, select, column, populate_result_map, asfrom, column_clause_args, name=None, proxy_name=None, fallback_label_name=None, within_columns_clause=True, column_is_repeated=False, need_column_expressions=False, include_table=True):
    """produce labeled columns present in a select()."""
    impl = column.type.dialect_impl(self.dialect)
    if impl._has_column_expression and (need_column_expressions or populate_result_map):
        col_expr = impl.column_expression(column)
    else:
        col_expr = column
    if populate_result_map:
        add_to_result_map = self._add_to_result_map
        if column_is_repeated:
            _add_to_result_map = add_to_result_map

            def add_to_result_map(keyname, name, objects, type_):
                _add_to_result_map(keyname, name, (), type_)
        elif col_expr is not column:
            _add_to_result_map = add_to_result_map

            def add_to_result_map(keyname, name, objects, type_):
                _add_to_result_map(keyname, name, (column,) + objects, type_)
    else:
        add_to_result_map = None
    assert within_columns_clause, '_label_select_column is only relevant within the columns clause of a SELECT or RETURNING'
    if isinstance(column, elements.Label):
        if col_expr is not column:
            result_expr = _CompileLabel(col_expr, column.name, alt_names=(column.element,))
        else:
            result_expr = col_expr
    elif name:
        assert proxy_name is not None, "proxy_name is required if 'name' is passed"
        result_expr = _CompileLabel(col_expr, name, alt_names=(proxy_name, column._tq_label))
    else:
        if col_expr is not column:
            render_with_label = True
        elif isinstance(column, elements.ColumnClause):
            render_with_label = asfrom and (not column.is_literal) and (column.table is not None)
        elif isinstance(column, elements.TextClause):
            render_with_label = False
        elif isinstance(column, elements.UnaryExpression):
            render_with_label = column.wraps_column_expression or asfrom
        elif not isinstance(column, elements.NamedColumn) and column._non_anon_label is None:
            render_with_label = True
        else:
            render_with_label = False
        if render_with_label:
            if not fallback_label_name:
                assert not column_is_repeated
                fallback_label_name = column._anon_name_label
            fallback_label_name = elements._truncated_label(fallback_label_name) if not isinstance(fallback_label_name, elements._truncated_label) else fallback_label_name
            result_expr = _CompileLabel(col_expr, fallback_label_name, alt_names=(proxy_name,))
        else:
            result_expr = col_expr
    column_clause_args.update(within_columns_clause=within_columns_clause, add_to_result_map=add_to_result_map, include_table=include_table)
    return result_expr._compiler_dispatch(self, **column_clause_args)