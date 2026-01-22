from __future__ import annotations
import collections
from enum import Enum
import itertools
from typing import AbstractSet
from typing import Any as TODO_Any
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import cache_key
from . import coercions
from . import operators
from . import roles
from . import traversals
from . import type_api
from . import visitors
from ._typing import _ColumnsClauseArgument
from ._typing import _no_kw
from ._typing import _TP
from ._typing import is_column_element
from ._typing import is_select_statement
from ._typing import is_subquery
from ._typing import is_table
from ._typing import is_text_clause
from .annotation import Annotated
from .annotation import SupportsCloneAnnotations
from .base import _clone
from .base import _cloned_difference
from .base import _cloned_intersection
from .base import _entity_namespace_key
from .base import _EntityNamespace
from .base import _expand_cloned
from .base import _from_objects
from .base import _generative
from .base import _never_select_column
from .base import _NoArg
from .base import _select_iterables
from .base import CacheableOptions
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DedupeColumnCollection
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .base import HasMemoized
from .base import Immutable
from .coercions import _document_text_coercion
from .elements import _anonymous_label
from .elements import BindParameter
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ClauseList
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import DQLDMLClauseElement
from .elements import GroupedElement
from .elements import literal_column
from .elements import TableValuedColumn
from .elements import UnaryExpression
from .operators import OperatorType
from .sqltypes import NULLTYPE
from .visitors import _TraverseInternalsType
from .visitors import InternalTraversal
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
@CompileState.plugin_for('default', 'select')
class SelectState(util.MemoizedSlots, CompileState):
    __slots__ = ('from_clauses', 'froms', 'columns_plus_names', '_label_resolve_dict')
    if TYPE_CHECKING:
        default_select_compile_options: CacheableOptions
    else:

        class default_select_compile_options(CacheableOptions):
            _cache_key_traversal = []
    if TYPE_CHECKING:

        @classmethod
        def get_plugin_class(cls, statement: Executable) -> Type[SelectState]:
            ...

    def __init__(self, statement: Select[Any], compiler: Optional[SQLCompiler], **kw: Any):
        self.statement = statement
        self.from_clauses = statement._from_obj
        for memoized_entities in statement._memoized_select_entities:
            self._setup_joins(memoized_entities._setup_joins, memoized_entities._raw_columns)
        if statement._setup_joins:
            self._setup_joins(statement._setup_joins, statement._raw_columns)
        self.froms = self._get_froms(statement)
        self.columns_plus_names = statement._generate_columns_plus_names(True)

    @classmethod
    def _plugin_not_implemented(cls) -> NoReturn:
        raise NotImplementedError('The default SELECT construct without plugins does not implement this method.')

    @classmethod
    def get_column_descriptions(cls, statement: Select[Any]) -> List[Dict[str, Any]]:
        return [{'name': name, 'type': element.type, 'expr': element} for _, name, _, element, _ in statement._generate_columns_plus_names(False)]

    @classmethod
    def from_statement(cls, statement: Select[Any], from_statement: roles.ReturnsRowsRole) -> ExecutableReturnsRows:
        cls._plugin_not_implemented()

    @classmethod
    def get_columns_clause_froms(cls, statement: Select[Any]) -> List[FromClause]:
        return cls._normalize_froms(itertools.chain.from_iterable((element._from_objects for element in statement._raw_columns)))

    @classmethod
    def _column_naming_convention(cls, label_style: SelectLabelStyle) -> _LabelConventionCallable:
        table_qualified = label_style is LABEL_STYLE_TABLENAME_PLUS_COL
        dedupe = label_style is not LABEL_STYLE_NONE
        pa = prefix_anon_map()
        names = set()

        def go(c: Union[ColumnElement[Any], TextClause], col_name: Optional[str]=None) -> Optional[str]:
            if is_text_clause(c):
                return None
            elif TYPE_CHECKING:
                assert is_column_element(c)
            if not dedupe:
                name = c._proxy_key
                if name is None:
                    name = '_no_label'
                return name
            name = c._tq_key_label if table_qualified else c._proxy_key
            if name is None:
                name = '_no_label'
                if name in names:
                    return c._anon_label(name) % pa
                else:
                    names.add(name)
                    return name
            elif name in names:
                return c._anon_tq_key_label % pa if table_qualified else c._anon_key_label % pa
            else:
                names.add(name)
                return name
        return go

    def _get_froms(self, statement: Select[Any]) -> List[FromClause]:
        ambiguous_table_name_map: _AmbiguousTableNameMap
        self._ambiguous_table_name_map = ambiguous_table_name_map = {}
        return self._normalize_froms(itertools.chain(self.from_clauses, itertools.chain.from_iterable([element._from_objects for element in statement._raw_columns]), itertools.chain.from_iterable([element._from_objects for element in statement._where_criteria])), check_statement=statement, ambiguous_table_name_map=ambiguous_table_name_map)

    @classmethod
    def _normalize_froms(cls, iterable_of_froms: Iterable[FromClause], check_statement: Optional[Select[Any]]=None, ambiguous_table_name_map: Optional[_AmbiguousTableNameMap]=None) -> List[FromClause]:
        """given an iterable of things to select FROM, reduce them to what
        would actually render in the FROM clause of a SELECT.

        This does the job of checking for JOINs, tables, etc. that are in fact
        overlapping due to cloning, adaption, present in overlapping joins,
        etc.

        """
        seen: Set[FromClause] = set()
        froms: List[FromClause] = []
        for item in iterable_of_froms:
            if is_subquery(item) and item.element is check_statement:
                raise exc.InvalidRequestError('select() construct refers to itself as a FROM')
            if not seen.intersection(item._cloned_set):
                froms.append(item)
                seen.update(item._cloned_set)
        if froms:
            toremove = set(itertools.chain.from_iterable([_expand_cloned(f._hide_froms) for f in froms]))
            if toremove:
                froms = [f for f in froms if f not in toremove]
            if ambiguous_table_name_map is not None:
                ambiguous_table_name_map.update(((fr.name, _anonymous_label.safe_construct(hash(fr.name), fr.name)) for item in froms for fr in item._from_objects if is_table(fr) and fr.schema and (fr.name not in ambiguous_table_name_map)))
        return froms

    def _get_display_froms(self, explicit_correlate_froms: Optional[Sequence[FromClause]]=None, implicit_correlate_froms: Optional[Sequence[FromClause]]=None) -> List[FromClause]:
        """Return the full list of 'from' clauses to be displayed.

        Takes into account a set of existing froms which may be
        rendered in the FROM clause of enclosing selects; this Select
        may want to leave those absent if it is automatically
        correlating.

        """
        froms = self.froms
        if self.statement._correlate:
            to_correlate = self.statement._correlate
            if to_correlate:
                froms = [f for f in froms if f not in _cloned_intersection(_cloned_intersection(froms, explicit_correlate_froms or ()), to_correlate)]
        if self.statement._correlate_except is not None:
            froms = [f for f in froms if f not in _cloned_difference(_cloned_intersection(froms, explicit_correlate_froms or ()), self.statement._correlate_except)]
        if self.statement._auto_correlate and implicit_correlate_froms and (len(froms) > 1):
            froms = [f for f in froms if f not in _cloned_intersection(froms, implicit_correlate_froms)]
            if not len(froms):
                raise exc.InvalidRequestError("Select statement '%r' returned no FROM clauses due to auto-correlation; specify correlate(<tables>) to control correlation manually." % self.statement)
        return froms

    def _memoized_attr__label_resolve_dict(self) -> Tuple[Dict[str, ColumnElement[Any]], Dict[str, ColumnElement[Any]], Dict[str, ColumnElement[Any]]]:
        with_cols: Dict[str, ColumnElement[Any]] = {c._tq_label or c.key: c for c in self.statement._all_selected_columns if c._allow_label_resolve}
        only_froms: Dict[str, ColumnElement[Any]] = {c.key: c for c in _select_iterables(self.froms) if c._allow_label_resolve}
        only_cols: Dict[str, ColumnElement[Any]] = with_cols.copy()
        for key, value in only_froms.items():
            with_cols.setdefault(key, value)
        return (with_cols, only_froms, only_cols)

    @classmethod
    def determine_last_joined_entity(cls, stmt: Select[Any]) -> Optional[_JoinTargetElement]:
        if stmt._setup_joins:
            return stmt._setup_joins[-1][0]
        else:
            return None

    @classmethod
    def all_selected_columns(cls, statement: Select[Any]) -> _SelectIterable:
        return [c for c in _select_iterables(statement._raw_columns)]

    def _setup_joins(self, args: Tuple[_SetupJoinsElement, ...], raw_columns: List[_ColumnsClauseElement]) -> None:
        for right, onclause, left, flags in args:
            if TYPE_CHECKING:
                if onclause is not None:
                    assert isinstance(onclause, ColumnElement)
            isouter = flags['isouter']
            full = flags['full']
            if left is None:
                left, replace_from_obj_index = self._join_determine_implicit_left_side(raw_columns, left, right, onclause)
            else:
                replace_from_obj_index = self._join_place_explicit_left_side(left)
            if TYPE_CHECKING:
                assert isinstance(right, FromClause)
                if onclause is not None:
                    assert isinstance(onclause, ColumnElement)
            if replace_from_obj_index is not None:
                left_clause = self.from_clauses[replace_from_obj_index]
                self.from_clauses = self.from_clauses[:replace_from_obj_index] + (Join(left_clause, right, onclause, isouter=isouter, full=full),) + self.from_clauses[replace_from_obj_index + 1:]
            else:
                assert left is not None
                self.from_clauses = self.from_clauses + (Join(left, right, onclause, isouter=isouter, full=full),)

    @util.preload_module('sqlalchemy.sql.util')
    def _join_determine_implicit_left_side(self, raw_columns: List[_ColumnsClauseElement], left: Optional[FromClause], right: _JoinTargetElement, onclause: Optional[ColumnElement[Any]]) -> Tuple[Optional[FromClause], Optional[int]]:
        """When join conditions don't express the left side explicitly,
        determine if an existing FROM or entity in this query
        can serve as the left hand side.

        """
        sql_util = util.preloaded.sql_util
        replace_from_obj_index: Optional[int] = None
        from_clauses = self.from_clauses
        if from_clauses:
            indexes: List[int] = sql_util.find_left_clause_to_join_from(from_clauses, right, onclause)
            if len(indexes) == 1:
                replace_from_obj_index = indexes[0]
                left = from_clauses[replace_from_obj_index]
        else:
            potential = {}
            statement = self.statement
            for from_clause in itertools.chain(itertools.chain.from_iterable([element._from_objects for element in raw_columns]), itertools.chain.from_iterable([element._from_objects for element in statement._where_criteria])):
                potential[from_clause] = ()
            all_clauses = list(potential.keys())
            indexes = sql_util.find_left_clause_to_join_from(all_clauses, right, onclause)
            if len(indexes) == 1:
                left = all_clauses[indexes[0]]
        if len(indexes) > 1:
            raise exc.InvalidRequestError("Can't determine which FROM clause to join from, there are multiple FROMS which can join to this entity. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity.")
        elif not indexes:
            raise exc.InvalidRequestError("Don't know how to join to %r. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity." % (right,))
        return (left, replace_from_obj_index)

    @util.preload_module('sqlalchemy.sql.util')
    def _join_place_explicit_left_side(self, left: FromClause) -> Optional[int]:
        replace_from_obj_index: Optional[int] = None
        sql_util = util.preloaded.sql_util
        from_clauses = list(self.statement._iterate_from_elements())
        if from_clauses:
            indexes: List[int] = sql_util.find_left_clause_that_matches_given(self.from_clauses, left)
        else:
            indexes = []
        if len(indexes) > 1:
            raise exc.InvalidRequestError("Can't identify which entity in which to assign the left side of this join.   Please use a more specific ON clause.")
        if indexes:
            replace_from_obj_index = indexes[0]
        return replace_from_obj_index