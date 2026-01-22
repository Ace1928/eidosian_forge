from __future__ import annotations
import collections
import itertools
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import interfaces
from . import loading
from . import path_registry
from . import properties
from . import query
from . import relationships
from . import unitofwork
from . import util as orm_util
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import ATTR_WAS_SET
from .base import LoaderCallableStatus
from .base import PASSIVE_OFF
from .base import PassiveFlag
from .context import _column_descriptions
from .context import ORMCompileState
from .context import ORMSelectCompileState
from .context import QueryContext
from .interfaces import LoaderStrategy
from .interfaces import StrategizedProperty
from .session import _state_session
from .state import InstanceState
from .strategy_options import Load
from .util import _none_set
from .util import AliasedClass
from .. import event
from .. import exc as sa_exc
from .. import inspect
from .. import log
from .. import sql
from .. import util
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
@log.class_logger
@relationships.RelationshipProperty.strategy_for(lazy='subquery')
class SubqueryLoader(PostLoader):
    __slots__ = ('join_depth',)

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy((('lazy', 'select'),)).init_class_attribute(mapper)

    def _get_leftmost(self, orig_query_entity_index, subq_path, current_compile_state, is_root):
        given_subq_path = subq_path
        subq_path = subq_path.path
        subq_mapper = orm_util._class_to_mapper(subq_path[0])
        if self.parent.isa(subq_mapper) and self.parent_property is subq_path[1]:
            leftmost_mapper, leftmost_prop = (self.parent, self.parent_property)
        else:
            leftmost_mapper, leftmost_prop = (subq_mapper, subq_path[1])
        if is_root:
            new_subq_path = current_compile_state._entities[orig_query_entity_index].entity_zero._path_registry[leftmost_prop]
            additional = len(subq_path) - len(new_subq_path)
            if additional:
                new_subq_path += path_registry.PathRegistry.coerce(subq_path[-additional:])
        else:
            new_subq_path = given_subq_path
        leftmost_cols = leftmost_prop.local_columns
        leftmost_attr = [getattr(new_subq_path.path[0].entity, leftmost_mapper._columntoproperty[c].key) for c in leftmost_cols]
        return (leftmost_mapper, leftmost_attr, leftmost_prop, new_subq_path)

    def _generate_from_original_query(self, orig_compile_state, orig_query, leftmost_mapper, leftmost_attr, leftmost_relationship, orig_entity):
        q = orig_query._clone().correlate(None)
        q2 = query.Query.__new__(query.Query)
        q2.__dict__.update(q.__dict__)
        q = q2
        if not q._from_obj:
            q._enable_assertions = False
            q.select_from.non_generative(q, *{ent['entity'] for ent in _column_descriptions(orig_query, compile_state=orig_compile_state) if ent['entity'] is not None})
        target_cols = orig_compile_state._adapt_col_list([sql.coercions.expect(sql.roles.ColumnsClauseRole, o) for o in leftmost_attr], orig_compile_state._get_current_adapter())
        q._raw_columns = target_cols
        distinct_target_key = leftmost_relationship.distinct_target_key
        if distinct_target_key is True:
            q._distinct = True
        elif distinct_target_key is None:
            for t in {c.table for c in target_cols}:
                if not set(target_cols).issuperset(t.primary_key):
                    q._distinct = True
                    break
        if not q._has_row_limiting_clause:
            q._order_by_clauses = ()
        if q._distinct is True and q._order_by_clauses:
            to_add = sql_util.expand_column_list_from_order_by(target_cols, q._order_by_clauses)
            if to_add:
                q._set_entities(target_cols + to_add)
        embed_q = q.set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL).subquery()
        left_alias = orm_util.AliasedClass(leftmost_mapper, embed_q, use_mapper_path=True)
        return left_alias

    def _prep_for_joins(self, left_alias, subq_path):
        to_join = []
        pairs = list(subq_path.pairs())
        for i, (mapper, prop) in enumerate(pairs):
            if i > 0:
                prev_mapper = pairs[i - 1][1].mapper
                to_append = prev_mapper if prev_mapper.isa(mapper) else mapper
            else:
                to_append = mapper
            to_join.append((to_append, prop.key))
        if len(to_join) < 2:
            parent_alias = left_alias
        else:
            info = inspect(to_join[-1][0])
            if info.is_aliased_class:
                parent_alias = info.entity
            else:
                parent_alias = orm_util.AliasedClass(info.entity, use_mapper_path=True)
        local_cols = self.parent_property.local_columns
        local_attr = [getattr(parent_alias, self.parent._columntoproperty[c].key) for c in local_cols]
        return (to_join, local_attr, parent_alias)

    def _apply_joins(self, q, to_join, left_alias, parent_alias, effective_entity):
        ltj = len(to_join)
        if ltj == 1:
            to_join = [getattr(left_alias, to_join[0][1]).of_type(effective_entity)]
        elif ltj == 2:
            to_join = [getattr(left_alias, to_join[0][1]).of_type(parent_alias), getattr(parent_alias, to_join[-1][1]).of_type(effective_entity)]
        elif ltj > 2:
            middle = [(orm_util.AliasedClass(item[0]) if not inspect(item[0]).is_aliased_class else item[0].entity, item[1]) for item in to_join[1:-1]]
            inner = []
            while middle:
                item = middle.pop(0)
                attr = getattr(item[0], item[1])
                if middle:
                    attr = attr.of_type(middle[0][0])
                else:
                    attr = attr.of_type(parent_alias)
                inner.append(attr)
            to_join = [getattr(left_alias, to_join[0][1]).of_type(inner[0].parent)] + inner + [getattr(parent_alias, to_join[-1][1]).of_type(effective_entity)]
        for attr in to_join:
            q = q.join(attr)
        return q

    def _setup_options(self, context, q, subq_path, rewritten_path, orig_query, effective_entity, loadopt):
        new_options = orig_query._with_options
        if loadopt and loadopt._extra_criteria:
            new_options += (orm_util.LoaderCriteriaOption(self.entity, loadopt._generate_extra_criteria(context)),)
        q = q._with_current_path(rewritten_path)
        q = q.options(*new_options)
        return q

    def _setup_outermost_orderby(self, q):
        if self.parent_property.order_by:

            def _setup_outermost_orderby(compile_context):
                compile_context.eager_order_by += tuple(util.to_list(self.parent_property.order_by))
            q = q._add_context_option(_setup_outermost_orderby, self.parent_property)
        return q

    class _SubqCollections:
        """Given a :class:`_query.Query` used to emit the "subquery load",
        provide a load interface that executes the query at the
        first moment a value is needed.

        """
        __slots__ = ('session', 'execution_options', 'load_options', 'params', 'subq', '_data')

        def __init__(self, context, subq):
            self.session = context.session
            self.execution_options = context.execution_options
            self.load_options = context.load_options
            self.params = context.params or {}
            self.subq = subq
            self._data = None

        def get(self, key, default):
            if self._data is None:
                self._load()
            return self._data.get(key, default)

        def _load(self):
            self._data = collections.defaultdict(list)
            q = self.subq
            assert q.session is None
            q = q.with_session(self.session)
            if self.load_options._populate_existing:
                q = q.populate_existing()
            rows = list(q.params(self.params))
            for k, v in itertools.groupby(rows, lambda x: x[1:]):
                self._data[k].extend((vv[0] for vv in v))

        def loader(self, state, dict_, row):
            if self._data is None:
                self._load()

    def _setup_query_from_rowproc(self, context, query_entity, path, entity, loadopt, adapter):
        compile_state = context.compile_state
        if not compile_state.compile_options._enable_eagerloads or compile_state.compile_options._for_refresh_state:
            return
        orig_query_entity_index = compile_state._entities.index(query_entity)
        context.loaders_require_buffering = True
        path = path[self.parent_property]
        with_poly_entity = path.get(compile_state.attributes, 'path_with_polymorphic', None)
        if with_poly_entity is not None:
            effective_entity = with_poly_entity
        else:
            effective_entity = self.entity
        subq_path, rewritten_path = context.query._execution_options.get(('subquery_paths', None), (orm_util.PathRegistry.root, orm_util.PathRegistry.root))
        is_root = subq_path is orm_util.PathRegistry.root
        subq_path = subq_path + path
        rewritten_path = rewritten_path + path
        orig_query = context.query._execution_options.get(('orig_query', SubqueryLoader), context.query)
        compile_state_cls = ORMCompileState._get_plugin_class_for_plugin(orig_query, 'orm')
        if orig_query._is_lambda_element:
            if context.load_options._lazy_loaded_from is None:
                util.warn('subqueryloader for "%s" must invoke lambda callable at %r in order to produce a new query, decreasing the efficiency of caching for this statement.  Consider using selectinload() for more effective full-lambda caching' % (self, orig_query))
            orig_query = orig_query._resolved
        orig_compile_state = compile_state_cls._create_entities_collection(orig_query, legacy=False)
        leftmost_mapper, leftmost_attr, leftmost_relationship, rewritten_path = self._get_leftmost(orig_query_entity_index, rewritten_path, orig_compile_state, is_root)
        left_alias = self._generate_from_original_query(orig_compile_state, orig_query, leftmost_mapper, leftmost_attr, leftmost_relationship, entity)
        q = query.Query(effective_entity)
        q._execution_options = context.query._execution_options.merge_with(context.execution_options, {('orig_query', SubqueryLoader): orig_query, ('subquery_paths', None): (subq_path, rewritten_path)})
        q = q._set_enable_single_crit(False)
        to_join, local_attr, parent_alias = self._prep_for_joins(left_alias, subq_path)
        q = q.add_columns(*local_attr)
        q = self._apply_joins(q, to_join, left_alias, parent_alias, effective_entity)
        q = self._setup_options(context, q, subq_path, rewritten_path, orig_query, effective_entity, loadopt)
        q = self._setup_outermost_orderby(q)
        return q

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        if context.refresh_state:
            return self._immediateload_create_row_processor(context, query_entity, path, loadopt, mapper, result, adapter, populators)
        _, run_loader, _, _ = self._setup_for_recursion(context, path, loadopt, self.join_depth)
        if not run_loader:
            return
        if not isinstance(context.compile_state, ORMSelectCompileState):
            return
        if not self.parent.class_manager[self.key].impl.supports_population:
            raise sa_exc.InvalidRequestError("'%s' does not support object population - eager loading cannot be applied." % self)
        if len(path) == 1:
            if not orm_util._entity_isa(query_entity.entity_zero, self.parent):
                return
        elif not orm_util._entity_isa(path[-1], self.parent):
            return
        subq = self._setup_query_from_rowproc(context, query_entity, path, path[-1], loadopt, adapter)
        if subq is None:
            return
        assert subq.session is None
        path = path[self.parent_property]
        local_cols = self.parent_property.local_columns
        collections = path.get(context.attributes, 'collections')
        if collections is None:
            collections = self._SubqCollections(context, subq)
            path.set(context.attributes, 'collections', collections)
        if adapter:
            local_cols = [adapter.columns[c] for c in local_cols]
        if self.uselist:
            self._create_collection_loader(context, result, collections, local_cols, populators)
        else:
            self._create_scalar_loader(context, result, collections, local_cols, populators)

    def _create_collection_loader(self, context, result, collections, local_cols, populators):
        tuple_getter = result._tuple_getter(local_cols)

        def load_collection_from_subq(state, dict_, row):
            collection = collections.get(tuple_getter(row), ())
            state.get_impl(self.key).set_committed_value(state, dict_, collection)

        def load_collection_from_subq_existing_row(state, dict_, row):
            if self.key not in dict_:
                load_collection_from_subq(state, dict_, row)
        populators['new'].append((self.key, load_collection_from_subq))
        populators['existing'].append((self.key, load_collection_from_subq_existing_row))
        if context.invoke_all_eagers:
            populators['eager'].append((self.key, collections.loader))

    def _create_scalar_loader(self, context, result, collections, local_cols, populators):
        tuple_getter = result._tuple_getter(local_cols)

        def load_scalar_from_subq(state, dict_, row):
            collection = collections.get(tuple_getter(row), (None,))
            if len(collection) > 1:
                util.warn("Multiple rows returned with uselist=False for eagerly-loaded attribute '%s' " % self)
            scalar = collection[0]
            state.get_impl(self.key).set_committed_value(state, dict_, scalar)

        def load_scalar_from_subq_existing_row(state, dict_, row):
            if self.key not in dict_:
                load_scalar_from_subq(state, dict_, row)
        populators['new'].append((self.key, load_scalar_from_subq))
        populators['existing'].append((self.key, load_scalar_from_subq_existing_row))
        if context.invoke_all_eagers:
            populators['eager'].append((self.key, collections.loader))