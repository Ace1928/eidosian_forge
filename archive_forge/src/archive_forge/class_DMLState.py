from __future__ import annotations
import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard
class DMLState(CompileState):
    _no_parameters = True
    _dict_parameters: Optional[MutableMapping[_DMLColumnElement, Any]] = None
    _multi_parameters: Optional[List[MutableMapping[_DMLColumnElement, Any]]] = None
    _ordered_values: Optional[List[Tuple[_DMLColumnElement, Any]]] = None
    _parameter_ordering: Optional[List[_DMLColumnElement]] = None
    _primary_table: FromClause
    _supports_implicit_returning = True
    isupdate = False
    isdelete = False
    isinsert = False
    statement: UpdateBase

    def __init__(self, statement: UpdateBase, compiler: SQLCompiler, **kw: Any):
        raise NotImplementedError()

    @classmethod
    def get_entity_description(cls, statement: UpdateBase) -> Dict[str, Any]:
        return {'name': statement.table.name if is_named_from_clause(statement.table) else None, 'table': statement.table}

    @classmethod
    def get_returning_column_descriptions(cls, statement: UpdateBase) -> List[Dict[str, Any]]:
        return [{'name': c.key, 'type': c.type, 'expr': c} for c in statement._all_selected_columns]

    @property
    def dml_table(self) -> _DMLTableElement:
        return self.statement.table
    if TYPE_CHECKING:

        @classmethod
        def get_plugin_class(cls, statement: Executable) -> Type[DMLState]:
            ...

    @classmethod
    def _get_multi_crud_kv_pairs(cls, statement: UpdateBase, multi_kv_iterator: Iterable[Dict[_DMLColumnArgument, Any]]) -> List[Dict[_DMLColumnElement, Any]]:
        return [{coercions.expect(roles.DMLColumnRole, k): v for k, v in mapping.items()} for mapping in multi_kv_iterator]

    @classmethod
    def _get_crud_kv_pairs(cls, statement: UpdateBase, kv_iterator: Iterable[Tuple[_DMLColumnArgument, Any]], needs_to_be_cacheable: bool) -> List[Tuple[_DMLColumnElement, Any]]:
        return [(coercions.expect(roles.DMLColumnRole, k), v if not needs_to_be_cacheable else coercions.expect(roles.ExpressionElementRole, v, type_=NullType(), is_crud=True)) for k, v in kv_iterator]

    def _make_extra_froms(self, statement: DMLWhereBase) -> Tuple[FromClause, List[FromClause]]:
        froms: List[FromClause] = []
        all_tables = list(sql_util.tables_from_leftmost(statement.table))
        primary_table = all_tables[0]
        seen = {primary_table}
        consider = statement._where_criteria
        if self._dict_parameters:
            consider += tuple(self._dict_parameters.values())
        for crit in consider:
            for item in _from_objects(crit):
                if not seen.intersection(item._cloned_set):
                    froms.append(item)
                seen.update(item._cloned_set)
        froms.extend(all_tables[1:])
        return (primary_table, froms)

    def _process_values(self, statement: ValuesBase) -> None:
        if self._no_parameters:
            self._dict_parameters = statement._values
            self._no_parameters = False

    def _process_select_values(self, statement: ValuesBase) -> None:
        assert statement._select_names is not None
        parameters: MutableMapping[_DMLColumnElement, Any] = {name: Null() for name in statement._select_names}
        if self._no_parameters:
            self._no_parameters = False
            self._dict_parameters = parameters
        else:
            assert False, 'This statement already has parameters'

    def _no_multi_values_supported(self, statement: ValuesBase) -> NoReturn:
        raise exc.InvalidRequestError('%s construct does not support multiple parameter sets.' % statement.__visit_name__.upper())

    def _cant_mix_formats_error(self) -> NoReturn:
        raise exc.InvalidRequestError("Can't mix single and multiple VALUES formats in one INSERT statement; one style appends to a list while the other replaces values, so the intent is ambiguous.")