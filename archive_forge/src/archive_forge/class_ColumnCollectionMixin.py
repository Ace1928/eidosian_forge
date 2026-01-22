from __future__ import annotations
from abc import ABC
import collections
from enum import Enum
import operator
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence as _typing_Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import ddl
from . import roles
from . import type_api
from . import visitors
from .base import _DefaultDescriptionTuple
from .base import _NoneName
from .base import _SentinelColumnCharacterization
from .base import _SentinelDefaultCharacterization
from .base import DedupeColumnCollection
from .base import DialectKWArgs
from .base import Executable
from .base import SchemaEventTarget as SchemaEventTarget
from .coercions import _document_text_coercion
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import quoted_name
from .elements import TextClause
from .selectable import TableClause
from .type_api import to_instance
from .visitors import ExternallyTraversible
from .visitors import InternalTraversal
from .. import event
from .. import exc
from .. import inspection
from .. import util
from ..util import HasMemoized
from ..util.typing import Final
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict
from ..util.typing import TypeGuard
class ColumnCollectionMixin:
    """A :class:`_expression.ColumnCollection` of :class:`_schema.Column`
    objects.

    This collection represents the columns which are referred to by
    this object.

    """
    _columns: DedupeColumnCollection[Column[Any]]
    _allow_multiple_tables = False
    _pending_colargs: List[Optional[Union[str, Column[Any]]]]
    if TYPE_CHECKING:

        def _set_parent_with_dispatch(self, parent: SchemaEventTarget, **kw: Any) -> None:
            ...

    def __init__(self, *columns: _DDLColumnArgument, _autoattach: bool=True, _column_flag: bool=False, _gather_expressions: Optional[List[Union[str, ColumnElement[Any]]]]=None) -> None:
        self._column_flag = _column_flag
        self._columns = DedupeColumnCollection()
        processed_expressions: Optional[List[Union[ColumnElement[Any], str]]] = _gather_expressions
        if processed_expressions is not None:
            self._pending_colargs = []
            for expr, _, _, add_element in coercions.expect_col_expression_collection(roles.DDLConstraintColumnRole, columns):
                self._pending_colargs.append(add_element)
                processed_expressions.append(expr)
        else:
            self._pending_colargs = [coercions.expect(roles.DDLConstraintColumnRole, column) for column in columns]
        if _autoattach and self._pending_colargs:
            self._check_attach()

    def _check_attach(self, evt: bool=False) -> None:
        col_objs = [c for c in self._pending_colargs if isinstance(c, Column)]
        cols_w_table = [c for c in col_objs if isinstance(c.table, Table)]
        cols_wo_table = set(col_objs).difference(cols_w_table)
        if cols_wo_table:
            assert not evt, 'Should not reach here on event call'
            has_string_cols = {c for c in self._pending_colargs if c is not None}.difference(col_objs)
            if not has_string_cols:

                def _col_attached(column: Column[Any], table: Table) -> None:
                    if isinstance(table, Table):
                        cols_wo_table.discard(column)
                        if not cols_wo_table:
                            self._check_attach(evt=True)
                self._cols_wo_table = cols_wo_table
                for col in cols_wo_table:
                    col._on_table_attach(_col_attached)
                return
        columns = cols_w_table
        tables = {c.table for c in columns}
        if len(tables) == 1:
            self._set_parent_with_dispatch(tables.pop())
        elif len(tables) > 1 and (not self._allow_multiple_tables):
            table = columns[0].table
            others = [c for c in columns[1:] if c.table is not table]
            if others:
                other_str = ', '.join(("'%s'" % c for c in others))
                raise exc.ArgumentError(f"Column(s) {other_str} are not part of table '{table.description}'.")

    @util.ro_memoized_property
    def columns(self) -> ReadOnlyColumnCollection[str, Column[Any]]:
        return self._columns.as_readonly()

    @util.ro_memoized_property
    def c(self) -> ReadOnlyColumnCollection[str, Column[Any]]:
        return self._columns.as_readonly()

    def _col_expressions(self, parent: Union[Table, Column[Any]]) -> List[Optional[Column[Any]]]:
        if isinstance(parent, Column):
            result: List[Optional[Column[Any]]] = [c for c in self._pending_colargs if isinstance(c, Column)]
            assert len(result) == len(self._pending_colargs)
            return result
        else:
            try:
                return [parent.c[col] if isinstance(col, str) else col for col in self._pending_colargs]
            except KeyError as ke:
                raise exc.ConstraintColumnNotFoundError(f"Can't create {self.__class__.__name__} on table '{parent.description}': no column named '{ke.args[0]}' is present.") from ke

    def _set_parent(self, parent: SchemaEventTarget, **kw: Any) -> None:
        assert isinstance(parent, (Table, Column))
        for col in self._col_expressions(parent):
            if col is not None:
                self._columns.add(col)