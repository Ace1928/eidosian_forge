from __future__ import annotations
import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal
@sql.base.CompileState.plugin_for('orm', 'select')
class ORMSelectCompileState(ORMCompileState, SelectState):
    _already_joined_edges = ()
    _memoized_entities = _EMPTY_DICT
    _from_obj_alias = None
    _has_mapper_entities = False
    _has_orm_entities = False
    multi_row_eager_loaders = False
    eager_adding_joins = False
    compound_eager_adapter = None
    correlate = None
    correlate_except = None
    _where_criteria = ()
    _having_criteria = ()

    @classmethod
    def create_for_statement(cls, statement: Union[Select, FromStatement], compiler: Optional[SQLCompiler], **kw: Any) -> ORMSelectCompileState:
        """compiler hook, we arrive here from compiler.visit_select() only."""
        self = cls.__new__(cls)
        if compiler is not None:
            toplevel = not compiler.stack
        else:
            toplevel = True
        select_statement = statement
        statement._compile_options = cls.default_compile_options.safe_merge(statement._compile_options)
        if select_statement._execution_options:
            self.select_statement = select_statement._clone()
            self.select_statement._execution_options = util.immutabledict()
        else:
            self.select_statement = select_statement
        self.for_statement = select_statement._compile_options._for_statement
        self.use_legacy_query_style = select_statement._compile_options._use_legacy_query_style
        self._entities = []
        self._primary_entity = None
        self._polymorphic_adapters = {}
        self.compile_options = select_statement._compile_options
        if not toplevel:
            self.compile_options += {'_enable_eagerloads': False, '_render_for_subquery': True}
        if self.use_legacy_query_style and self.select_statement._label_style is LABEL_STYLE_LEGACY_ORM:
            if not self.for_statement:
                self.label_style = LABEL_STYLE_TABLENAME_PLUS_COL
            else:
                self.label_style = LABEL_STYLE_DISAMBIGUATE_ONLY
        else:
            self.label_style = self.select_statement._label_style
        if select_statement._memoized_select_entities:
            self._memoized_entities = {memoized_entities: _QueryEntity.to_compile_state(self, memoized_entities._raw_columns, [], is_current_entities=False) for memoized_entities in select_statement._memoized_select_entities}
        self._label_convention = self._column_naming_convention(statement._label_style, self.use_legacy_query_style)
        _QueryEntity.to_compile_state(self, select_statement._raw_columns, self._entities, is_current_entities=True)
        self.current_path = select_statement._compile_options._current_path
        self.eager_order_by = ()
        self._init_global_attributes(select_statement, compiler, toplevel=toplevel, process_criteria_for_toplevel=False)
        if toplevel and (select_statement._with_options or select_statement._memoized_select_entities):
            for memoized_entities in select_statement._memoized_select_entities:
                for opt in memoized_entities._with_options:
                    if opt._is_compile_state:
                        opt.process_compile_state_replaced_entities(self, [ent for ent in self._memoized_entities[memoized_entities] if isinstance(ent, _MapperEntity)])
            for opt in self.select_statement._with_options:
                if opt._is_compile_state:
                    opt.process_compile_state(self)
        if select_statement._with_context_options:
            for fn, key in select_statement._with_context_options:
                fn(self)
        self.primary_columns = []
        self.secondary_columns = []
        self.dedupe_columns = set()
        self.eager_joins = {}
        self.extra_criteria_entities = {}
        self.create_eager_joins = []
        self._fallback_from_clauses = []
        self.from_clauses = self._normalize_froms((info.selectable for info in select_statement._from_obj))
        self._setup_for_generate()
        SelectState.__init__(self, self.statement, compiler, **kw)
        return self

    def _dump_option_struct(self):
        print('\n---------------------------------------------------\n')
        print(f'current path: {self.current_path}')
        for key in self.attributes:
            if isinstance(key, tuple) and key[0] == 'loader':
                print(f'\nLoader:           {PathRegistry.coerce(key[1])}')
                print(f'    {self.attributes[key]}')
                print(f'    {self.attributes[key].__dict__}')
            elif isinstance(key, tuple) and key[0] == 'path_with_polymorphic':
                print(f'\nWith Polymorphic: {PathRegistry.coerce(key[1])}')
                print(f'    {self.attributes[key]}')

    def _setup_for_generate(self):
        query = self.select_statement
        self.statement = None
        self._join_entities = ()
        if self.compile_options._set_base_alias:
            self._set_select_from_alias()
        for memoized_entities in query._memoized_select_entities:
            if memoized_entities._setup_joins:
                self._join(memoized_entities._setup_joins, self._memoized_entities[memoized_entities])
        if query._setup_joins:
            self._join(query._setup_joins, self._entities)
        current_adapter = self._get_current_adapter()
        if query._where_criteria:
            self._where_criteria = query._where_criteria
            if current_adapter:
                self._where_criteria = tuple((current_adapter(crit, True) for crit in self._where_criteria))
        self.order_by = self._adapt_col_list(query._order_by_clauses, current_adapter) if current_adapter and query._order_by_clauses not in (None, False) else query._order_by_clauses
        if query._having_criteria:
            self._having_criteria = tuple((current_adapter(crit, True) if current_adapter else crit for crit in query._having_criteria))
        self.group_by = self._adapt_col_list(util.flatten_iterator(query._group_by_clauses), current_adapter) if current_adapter and query._group_by_clauses not in (None, False) else query._group_by_clauses or None
        if self.eager_order_by:
            adapter = self.from_clauses[0]._target_adapter
            self.eager_order_by = adapter.copy_and_process(self.eager_order_by)
        if query._distinct_on:
            self.distinct_on = self._adapt_col_list(query._distinct_on, current_adapter)
        else:
            self.distinct_on = ()
        self.distinct = query._distinct
        if query._correlate:
            self.correlate = tuple(util.flatten_iterator((sql_util.surface_selectables(s) if s is not None else None for s in query._correlate)))
        elif query._correlate_except is not None:
            self.correlate_except = tuple(util.flatten_iterator((sql_util.surface_selectables(s) if s is not None else None for s in query._correlate_except)))
        elif not query._auto_correlate:
            self.correlate = (None,)
        self._for_update_arg = query._for_update_arg
        if self.compile_options._is_star and len(self._entities) != 1:
            raise sa_exc.CompileError("Can't generate ORM query that includes multiple expressions at the same time as '*'; query for '*' alone if present")
        for entity in self._entities:
            entity.setup_compile_state(self)
        for rec in self.create_eager_joins:
            strategy = rec[0]
            strategy(self, *rec[1:])
        if self.compile_options._enable_single_crit:
            self._adjust_for_extra_criteria()
        if not self.primary_columns:
            if self.compile_options._only_load_props:
                assert False, 'no columns were included in _only_load_props'
            raise sa_exc.InvalidRequestError('Query contains no columns with which to SELECT from.')
        if not self.from_clauses:
            self.from_clauses = list(self._fallback_from_clauses)
        if self.order_by is False:
            self.order_by = None
        if self.multi_row_eager_loaders and self.eager_adding_joins and self._should_nest_selectable:
            self.statement = self._compound_eager_statement()
        else:
            self.statement = self._simple_statement()
        if self.for_statement:
            ezero = self._mapper_zero()
            if ezero is not None:
                self.statement = self.statement._annotate({'deepentity': ezero})

    @classmethod
    def _create_entities_collection(cls, query, legacy):
        """Creates a partial ORMSelectCompileState that includes
        the full collection of _MapperEntity and other _QueryEntity objects.

        Supports a few remaining use cases that are pre-compilation
        but still need to gather some of the column  / adaption information.

        """
        self = cls.__new__(cls)
        self._entities = []
        self._primary_entity = None
        self._polymorphic_adapters = {}
        self._label_convention = self._column_naming_convention(query._label_style, legacy)
        _QueryEntity.to_compile_state(self, query._raw_columns, self._entities, is_current_entities=True)
        return self

    @classmethod
    def determine_last_joined_entity(cls, statement):
        setup_joins = statement._setup_joins
        return _determine_last_joined_entity(setup_joins, None)

    @classmethod
    def all_selected_columns(cls, statement):
        for element in statement._raw_columns:
            if element.is_selectable and 'entity_namespace' in element._annotations:
                ens = element._annotations['entity_namespace']
                if not ens.is_mapper and (not ens.is_aliased_class):
                    yield from _select_iterables([element])
                else:
                    yield from _select_iterables(ens._all_column_expressions)
            else:
                yield from _select_iterables([element])

    @classmethod
    def get_columns_clause_froms(cls, statement):
        return cls._normalize_froms(itertools.chain.from_iterable((element._from_objects if 'parententity' not in element._annotations else [element._annotations['parententity'].__clause_element__()] for element in statement._raw_columns)))

    @classmethod
    def from_statement(cls, statement, from_statement):
        from_statement = coercions.expect(roles.ReturnsRowsRole, from_statement, apply_propagate_attrs=statement)
        stmt = FromStatement(statement._raw_columns, from_statement)
        stmt.__dict__.update(_with_options=statement._with_options, _with_context_options=statement._with_context_options, _execution_options=statement._execution_options, _propagate_attrs=statement._propagate_attrs)
        return stmt

    def _set_select_from_alias(self):
        """used only for legacy Query cases"""
        query = self.select_statement
        assert self.compile_options._set_base_alias
        assert len(query._from_obj) == 1
        adapter = self._get_select_from_alias_from_obj(query._from_obj[0])
        if adapter:
            self.compile_options += {'_enable_single_crit': False}
            self._from_obj_alias = adapter

    def _get_select_from_alias_from_obj(self, from_obj):
        """used only for legacy Query cases"""
        info = from_obj
        if 'parententity' in info._annotations:
            info = info._annotations['parententity']
        if hasattr(info, 'mapper'):
            if not info.is_aliased_class:
                raise sa_exc.ArgumentError('A selectable (FromClause) instance is expected when the base alias is being set.')
            else:
                return info._adapter
        elif isinstance(info.selectable, sql.selectable.AliasedReturnsRows):
            equivs = self._all_equivs()
            assert info is info.selectable
            return ORMStatementAdapter(_TraceAdaptRole.LEGACY_SELECT_FROM_ALIAS, info.selectable, equivalents=equivs)
        else:
            return None

    def _mapper_zero(self):
        """return the Mapper associated with the first QueryEntity."""
        return self._entities[0].mapper

    def _entity_zero(self):
        """Return the 'entity' (mapper or AliasedClass) associated
        with the first QueryEntity, or alternatively the 'select from'
        entity if specified."""
        for ent in self.from_clauses:
            if 'parententity' in ent._annotations:
                return ent._annotations['parententity']
        for qent in self._entities:
            if qent.entity_zero:
                return qent.entity_zero
        return None

    def _only_full_mapper_zero(self, methname):
        if self._entities != [self._primary_entity]:
            raise sa_exc.InvalidRequestError('%s() can only be used against a single mapped class.' % methname)
        return self._primary_entity.entity_zero

    def _only_entity_zero(self, rationale=None):
        if len(self._entities) > 1:
            raise sa_exc.InvalidRequestError(rationale or 'This operation requires a Query against a single mapper.')
        return self._entity_zero()

    def _all_equivs(self):
        equivs = {}
        for memoized_entities in self._memoized_entities.values():
            for ent in [ent for ent in memoized_entities if isinstance(ent, _MapperEntity)]:
                equivs.update(ent.mapper._equivalent_columns)
        for ent in [ent for ent in self._entities if isinstance(ent, _MapperEntity)]:
            equivs.update(ent.mapper._equivalent_columns)
        return equivs

    def _compound_eager_statement(self):
        if self.order_by:
            unwrapped_order_by = [elem.element if isinstance(elem, sql.elements._label_reference) else elem for elem in self.order_by]
            order_by_col_expr = sql_util.expand_column_list_from_order_by(self.primary_columns, unwrapped_order_by)
        else:
            order_by_col_expr = []
            unwrapped_order_by = None
        inner = self._select_statement(self.primary_columns + [c for c in order_by_col_expr if c not in self.dedupe_columns], self.from_clauses, self._where_criteria, self._having_criteria, self.label_style, self.order_by, for_update=self._for_update_arg, hints=self.select_statement._hints, statement_hints=self.select_statement._statement_hints, correlate=self.correlate, correlate_except=self.correlate_except, **self._select_args)
        inner = inner.alias()
        equivs = self._all_equivs()
        self.compound_eager_adapter = ORMStatementAdapter(_TraceAdaptRole.COMPOUND_EAGER_STATEMENT, inner, equivalents=equivs)
        statement = future.select(*[inner] + self.secondary_columns)
        statement._label_style = self.label_style
        if self._for_update_arg is not None and self._for_update_arg.of is None:
            statement._for_update_arg = self._for_update_arg
        from_clause = inner
        for eager_join in self.eager_joins.values():
            from_clause = sql_util.splice_joins(from_clause, eager_join, eager_join.stop_on)
        statement.select_from.non_generative(statement, from_clause)
        if unwrapped_order_by:
            statement.order_by.non_generative(statement, *self.compound_eager_adapter.copy_and_process(unwrapped_order_by))
        statement.order_by.non_generative(statement, *self.eager_order_by)
        return statement

    def _simple_statement(self):
        statement = self._select_statement(self.primary_columns + self.secondary_columns, tuple(self.from_clauses) + tuple(self.eager_joins.values()), self._where_criteria, self._having_criteria, self.label_style, self.order_by, for_update=self._for_update_arg, hints=self.select_statement._hints, statement_hints=self.select_statement._statement_hints, correlate=self.correlate, correlate_except=self.correlate_except, **self._select_args)
        if self.eager_order_by:
            statement.order_by.non_generative(statement, *self.eager_order_by)
        return statement

    def _select_statement(self, raw_columns, from_obj, where_criteria, having_criteria, label_style, order_by, for_update, hints, statement_hints, correlate, correlate_except, limit_clause, offset_clause, fetch_clause, fetch_clause_options, distinct, distinct_on, prefixes, suffixes, group_by, independent_ctes, independent_ctes_opts):
        statement = Select._create_raw_select(_raw_columns=raw_columns, _from_obj=from_obj, _label_style=label_style)
        if where_criteria:
            statement._where_criteria = where_criteria
        if having_criteria:
            statement._having_criteria = having_criteria
        if order_by:
            statement._order_by_clauses += tuple(order_by)
        if distinct_on:
            statement.distinct.non_generative(statement, *distinct_on)
        elif distinct:
            statement.distinct.non_generative(statement)
        if group_by:
            statement._group_by_clauses += tuple(group_by)
        statement._limit_clause = limit_clause
        statement._offset_clause = offset_clause
        statement._fetch_clause = fetch_clause
        statement._fetch_clause_options = fetch_clause_options
        statement._independent_ctes = independent_ctes
        statement._independent_ctes_opts = independent_ctes_opts
        if prefixes:
            statement._prefixes = prefixes
        if suffixes:
            statement._suffixes = suffixes
        statement._for_update_arg = for_update
        if hints:
            statement._hints = hints
        if statement_hints:
            statement._statement_hints = statement_hints
        if correlate:
            statement.correlate.non_generative(statement, *correlate)
        if correlate_except is not None:
            statement.correlate_except.non_generative(statement, *correlate_except)
        return statement

    def _adapt_polymorphic_element(self, element):
        if 'parententity' in element._annotations:
            search = element._annotations['parententity']
            alias = self._polymorphic_adapters.get(search, None)
            if alias:
                return alias.adapt_clause(element)
        if isinstance(element, expression.FromClause):
            search = element
        elif hasattr(element, 'table'):
            search = element.table
        else:
            return None
        alias = self._polymorphic_adapters.get(search, None)
        if alias:
            return alias.adapt_clause(element)

    def _adapt_col_list(self, cols, current_adapter):
        if current_adapter:
            return [current_adapter(o, True) for o in cols]
        else:
            return cols

    def _get_current_adapter(self):
        adapters = []
        if self._from_obj_alias:
            adapters.append((True, self._from_obj_alias.replace))
        if self._polymorphic_adapters:
            adapters.append((False, self._adapt_polymorphic_element))
        if not adapters:
            return None

        def _adapt_clause(clause, as_filter):

            def replace(elem):
                is_orm_adapt = '_orm_adapt' in elem._annotations or 'parententity' in elem._annotations
                for always_adapt, adapter in adapters:
                    if is_orm_adapt or always_adapt:
                        e = adapter(elem)
                        if e is not None:
                            return e
            return visitors.replacement_traverse(clause, {}, replace)
        return _adapt_clause

    def _join(self, args, entities_collection):
        for right, onclause, from_, flags in args:
            isouter = flags['isouter']
            full = flags['full']
            right = inspect(right)
            if onclause is not None:
                onclause = inspect(onclause)
            if isinstance(right, interfaces.PropComparator):
                if onclause is not None:
                    raise sa_exc.InvalidRequestError("No 'on clause' argument may be passed when joining to a relationship path as a target")
                onclause = right
                right = None
            elif 'parententity' in right._annotations:
                right = right._annotations['parententity']
            if onclause is None:
                if not right.is_selectable and (not hasattr(right, 'mapper')):
                    raise sa_exc.ArgumentError('Expected mapped entity or selectable/table as join target')
            of_type = None
            if isinstance(onclause, interfaces.PropComparator):
                of_type = getattr(onclause, '_of_type', None)
                if right is None:
                    if of_type:
                        right = of_type
                    else:
                        right = onclause.property
                        try:
                            right = right.entity
                        except AttributeError as err:
                            raise sa_exc.ArgumentError('Join target %s does not refer to a mapped entity' % right) from err
                left = onclause._parententity
                prop = onclause.property
                if not isinstance(onclause, attributes.QueryableAttribute):
                    onclause = prop
                if (left, right, prop.key) in self._already_joined_edges:
                    continue
                if from_ is not None:
                    if from_ is not left and from_._annotations.get('parententity', None) is not left:
                        raise sa_exc.InvalidRequestError('explicit from clause %s does not match left side of relationship attribute %s' % (from_._annotations.get('parententity', from_), onclause))
            elif from_ is not None:
                prop = None
                left = from_
            else:
                prop = left = None
            self._join_left_to_right(entities_collection, left, right, onclause, prop, isouter, full)

    def _join_left_to_right(self, entities_collection, left, right, onclause, prop, outerjoin, full):
        """given raw "left", "right", "onclause" parameters consumed from
        a particular key within _join(), add a real ORMJoin object to
        our _from_obj list (or augment an existing one)

        """
        if left is None:
            assert prop is None
            left, replace_from_obj_index, use_entity_index = self._join_determine_implicit_left_side(entities_collection, left, right, onclause)
        else:
            replace_from_obj_index, use_entity_index = self._join_place_explicit_left_side(entities_collection, left)
        if left is right:
            raise sa_exc.InvalidRequestError("Can't construct a join from %s to %s, they are the same entity" % (left, right))
        r_info, right, onclause = self._join_check_and_adapt_right_side(left, right, onclause, prop)
        if not r_info.is_selectable:
            extra_criteria = self._get_extra_criteria(r_info)
        else:
            extra_criteria = ()
        if replace_from_obj_index is not None:
            left_clause = self.from_clauses[replace_from_obj_index]
            self.from_clauses = self.from_clauses[:replace_from_obj_index] + [_ORMJoin(left_clause, right, onclause, isouter=outerjoin, full=full, _extra_criteria=extra_criteria)] + self.from_clauses[replace_from_obj_index + 1:]
        else:
            if use_entity_index is not None:
                assert isinstance(entities_collection[use_entity_index], _MapperEntity)
                left_clause = entities_collection[use_entity_index].selectable
            else:
                left_clause = left
            self.from_clauses = self.from_clauses + [_ORMJoin(left_clause, r_info, onclause, isouter=outerjoin, full=full, _extra_criteria=extra_criteria)]

    def _join_determine_implicit_left_side(self, entities_collection, left, right, onclause):
        """When join conditions don't express the left side explicitly,
        determine if an existing FROM or entity in this query
        can serve as the left hand side.

        """
        r_info = inspect(right)
        replace_from_obj_index = use_entity_index = None
        if self.from_clauses:
            indexes = sql_util.find_left_clause_to_join_from(self.from_clauses, r_info.selectable, onclause)
            if len(indexes) == 1:
                replace_from_obj_index = indexes[0]
                left = self.from_clauses[replace_from_obj_index]
            elif len(indexes) > 1:
                raise sa_exc.InvalidRequestError("Can't determine which FROM clause to join from, there are multiple FROMS which can join to this entity. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity.")
            else:
                raise sa_exc.InvalidRequestError("Don't know how to join to %r. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity." % (right,))
        elif entities_collection:
            potential = {}
            for entity_index, ent in enumerate(entities_collection):
                entity = ent.entity_zero_or_selectable
                if entity is None:
                    continue
                ent_info = inspect(entity)
                if ent_info is r_info:
                    continue
                if isinstance(ent, _MapperEntity):
                    potential[ent.selectable] = (entity_index, entity)
                else:
                    potential[ent_info.selectable] = (None, entity)
            all_clauses = list(potential.keys())
            indexes = sql_util.find_left_clause_to_join_from(all_clauses, r_info.selectable, onclause)
            if len(indexes) == 1:
                use_entity_index, left = potential[all_clauses[indexes[0]]]
            elif len(indexes) > 1:
                raise sa_exc.InvalidRequestError("Can't determine which FROM clause to join from, there are multiple FROMS which can join to this entity. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity.")
            else:
                raise sa_exc.InvalidRequestError("Don't know how to join to %r. Please use the .select_from() method to establish an explicit left side, as well as providing an explicit ON clause if not present already to help resolve the ambiguity." % (right,))
        else:
            raise sa_exc.InvalidRequestError('No entities to join from; please use select_from() to establish the left entity/selectable of this join')
        return (left, replace_from_obj_index, use_entity_index)

    def _join_place_explicit_left_side(self, entities_collection, left):
        """When join conditions express a left side explicitly, determine
        where in our existing list of FROM clauses we should join towards,
        or if we need to make a new join, and if so is it from one of our
        existing entities.

        """
        replace_from_obj_index = use_entity_index = None
        l_info = inspect(left)
        if self.from_clauses:
            indexes = sql_util.find_left_clause_that_matches_given(self.from_clauses, l_info.selectable)
            if len(indexes) > 1:
                raise sa_exc.InvalidRequestError("Can't identify which entity in which to assign the left side of this join.   Please use a more specific ON clause.")
            if indexes:
                replace_from_obj_index = indexes[0]
        if replace_from_obj_index is None and entities_collection and hasattr(l_info, 'mapper'):
            for idx, ent in enumerate(entities_collection):
                if isinstance(ent, _MapperEntity) and ent.corresponds_to(left):
                    use_entity_index = idx
                    break
        return (replace_from_obj_index, use_entity_index)

    def _join_check_and_adapt_right_side(self, left, right, onclause, prop):
        """transform the "right" side of the join as well as the onclause
        according to polymorphic mapping translations, aliasing on the query
        or on the join, special cases where the right and left side have
        overlapping tables.

        """
        l_info = inspect(left)
        r_info = inspect(right)
        overlap = False
        right_mapper = getattr(r_info, 'mapper', None)
        if right_mapper and (right_mapper.with_polymorphic or isinstance(right_mapper.persist_selectable, expression.Join)):
            for from_obj in self.from_clauses or [l_info.selectable]:
                if sql_util.selectables_overlap(l_info.selectable, from_obj) and sql_util.selectables_overlap(from_obj, r_info.selectable):
                    overlap = True
                    break
        if overlap and l_info.selectable is r_info.selectable:
            raise sa_exc.InvalidRequestError("Can't join table/selectable '%s' to itself" % l_info.selectable)
        right_mapper, right_selectable, right_is_aliased = (getattr(r_info, 'mapper', None), r_info.selectable, getattr(r_info, 'is_aliased_class', False))
        if right_mapper and prop and (not right_mapper.common_parent(prop.mapper)):
            raise sa_exc.InvalidRequestError('Join target %s does not correspond to the right side of join condition %s' % (right, onclause))
        if hasattr(r_info, 'mapper'):
            self._join_entities += (r_info,)
        need_adapter = False
        if r_info.is_clause_element:
            if prop:
                right_mapper = prop.mapper
            if right_selectable._is_lateral:
                current_adapter = self._get_current_adapter()
                if current_adapter is not None:
                    right = current_adapter(right, True)
            elif prop:
                if not right_selectable.is_derived_from(right_mapper.persist_selectable):
                    raise sa_exc.InvalidRequestError("Selectable '%s' is not derived from '%s'" % (right_selectable.description, right_mapper.persist_selectable.description))
                if isinstance(right_selectable, expression.SelectBase):
                    right_selectable = coercions.expect(roles.FromClauseRole, right_selectable)
                    need_adapter = True
                right = AliasedClass(right_mapper, right_selectable)
                util.warn_deprecated('An alias is being generated automatically against joined entity %s for raw clauseelement, which is deprecated and will be removed in a later release. Use the aliased() construct explicitly, see the linked example.' % right_mapper, '1.4', code='xaj1')
        aliased_entity = right_mapper and (not right_is_aliased) and overlap
        if not need_adapter and aliased_entity:
            right = AliasedClass(right, flat=True)
            need_adapter = True
            util.warn('An alias is being generated automatically against joined entity %s due to overlapping tables.  This is a legacy pattern which may be deprecated in a later release.  Use the aliased(<entity>, flat=True) construct explicitly, see the linked example.' % right_mapper, code='xaj2')
        if need_adapter:
            assert right_mapper
            adapter = ORMAdapter(_TraceAdaptRole.DEPRECATED_JOIN_ADAPT_RIGHT_SIDE, inspect(right), equivalents=right_mapper._equivalent_columns)
            self._mapper_loads_polymorphically_with(right_mapper, adapter)
        elif not r_info.is_clause_element and (not right_is_aliased) and right_mapper._has_aliased_polymorphic_fromclause:
            self._mapper_loads_polymorphically_with(right_mapper, ORMAdapter(_TraceAdaptRole.WITH_POLYMORPHIC_ADAPTER_RIGHT_JOIN, right_mapper, selectable=right_mapper.selectable, equivalents=right_mapper._equivalent_columns))
        if isinstance(onclause, expression.ClauseElement):
            current_adapter = self._get_current_adapter()
            if current_adapter:
                onclause = current_adapter(onclause, True)
        if prop:
            self._already_joined_edges += ((left, right, prop.key),)
        return (inspect(right), right, onclause)

    @property
    def _select_args(self):
        return {'limit_clause': self.select_statement._limit_clause, 'offset_clause': self.select_statement._offset_clause, 'distinct': self.distinct, 'distinct_on': self.distinct_on, 'prefixes': self.select_statement._prefixes, 'suffixes': self.select_statement._suffixes, 'group_by': self.group_by or None, 'fetch_clause': self.select_statement._fetch_clause, 'fetch_clause_options': self.select_statement._fetch_clause_options, 'independent_ctes': self.select_statement._independent_ctes, 'independent_ctes_opts': self.select_statement._independent_ctes_opts}

    @property
    def _should_nest_selectable(self):
        kwargs = self._select_args
        return kwargs.get('limit_clause') is not None or kwargs.get('offset_clause') is not None or kwargs.get('distinct', False) or kwargs.get('distinct_on', ()) or kwargs.get('group_by', False)

    def _get_extra_criteria(self, ext_info):
        if ('additional_entity_criteria', ext_info.mapper) in self.global_attributes:
            return tuple((ae._resolve_where_criteria(ext_info) for ae in self.global_attributes['additional_entity_criteria', ext_info.mapper] if (ae.include_aliases or ae.entity is ext_info) and ae._should_include(self)))
        else:
            return ()

    def _adjust_for_extra_criteria(self):
        """Apply extra criteria filtering.

        For all distinct single-table-inheritance mappers represented in
        the columns clause of this query, as well as the "select from entity",
        add criterion to the WHERE
        clause of the given QueryContext such that only the appropriate
        subtypes are selected from the total results.

        Additionally, add WHERE criteria originating from LoaderCriteriaOptions
        associated with the global context.

        """
        for fromclause in self.from_clauses:
            ext_info = fromclause._annotations.get('parententity', None)
            if ext_info and (ext_info.mapper._single_table_criterion is not None or ('additional_entity_criteria', ext_info.mapper) in self.global_attributes) and (ext_info not in self.extra_criteria_entities):
                self.extra_criteria_entities[ext_info] = (ext_info, ext_info._adapter if ext_info.is_aliased_class else None)
        search = set(self.extra_criteria_entities.values())
        for ext_info, adapter in search:
            if ext_info in self._join_entities:
                continue
            single_crit = ext_info.mapper._single_table_criterion
            if self.compile_options._for_refresh_state:
                additional_entity_criteria = []
            else:
                additional_entity_criteria = self._get_extra_criteria(ext_info)
            if single_crit is not None:
                additional_entity_criteria += (single_crit,)
            current_adapter = self._get_current_adapter()
            for crit in additional_entity_criteria:
                if adapter:
                    crit = adapter.traverse(crit)
                if current_adapter:
                    crit = sql_util._deep_annotate(crit, {'_orm_adapt': True})
                    crit = current_adapter(crit, False)
                self._where_criteria += (crit,)