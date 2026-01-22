from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import path_registry
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import PassiveFlag
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .util import _none_set
from .util import state_str
from .. import exc as sa_exc
from .. import util
from ..engine import result_tuple
from ..engine.result import ChunkedIteratorResult
from ..engine.result import FrozenResult
from ..engine.result import SimpleResultMetaData
from ..sql import select
from ..sql import util as sql_util
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectState
from ..util import EMPTY_DICT
def _instance_processor(query_entity, mapper, context, result, path, adapter, only_load_props=None, refresh_state=None, polymorphic_discriminator=None, _polymorphic_from=None):
    """Produce a mapper level row processor callable
    which processes rows into mapped instances."""
    identity_class = mapper._identity_class
    compile_state = context.compile_state
    getter_key = ('getters', mapper)
    getters = path.get(compile_state.attributes, getter_key, None)
    if getters is None:
        props = mapper._prop_set
        if only_load_props is not None:
            props = props.intersection((mapper._props[k] for k in only_load_props))
        quick_populators = path.get(context.attributes, 'memoized_setups', EMPTY_DICT)
        todo = []
        cached_populators = {'new': [], 'quick': [], 'deferred': [], 'expire': [], 'existing': [], 'eager': []}
        if refresh_state is None:
            pk_cols = mapper.primary_key
            if adapter:
                pk_cols = [adapter.columns[c] for c in pk_cols]
            primary_key_getter = result._tuple_getter(pk_cols)
        else:
            primary_key_getter = None
        getters = {'cached_populators': cached_populators, 'todo': todo, 'primary_key_getter': primary_key_getter}
        for prop in props:
            if prop in quick_populators:
                col = quick_populators[prop]
                if col is _DEFER_FOR_STATE:
                    cached_populators['new'].append((prop.key, prop._deferred_column_loader))
                elif col is _SET_DEFERRED_EXPIRED:
                    cached_populators['expire'].append((prop.key, False))
                elif col is _RAISE_FOR_STATE:
                    cached_populators['new'].append((prop.key, prop._raise_column_loader))
                else:
                    getter = None
                    if adapter:
                        adapted_col = adapter.columns[col]
                        if adapted_col is not None:
                            getter = result._getter(adapted_col, False)
                    if not getter:
                        getter = result._getter(col, False)
                    if getter:
                        cached_populators['quick'].append((prop.key, getter))
                    else:
                        prop.create_row_processor(context, query_entity, path, mapper, result, adapter, cached_populators)
            else:
                todo.append(prop)
        path.set(compile_state.attributes, getter_key, getters)
    cached_populators = getters['cached_populators']
    populators = {key: list(value) for key, value in cached_populators.items()}
    for prop in getters['todo']:
        prop.create_row_processor(context, query_entity, path, mapper, result, adapter, populators)
    propagated_loader_options = context.propagated_loader_options
    load_path = context.compile_state.current_path + path if context.compile_state.current_path.path else path
    session_identity_map = context.session.identity_map
    populate_existing = context.populate_existing or mapper.always_refresh
    load_evt = bool(mapper.class_manager.dispatch.load)
    refresh_evt = bool(mapper.class_manager.dispatch.refresh)
    persistent_evt = bool(context.session.dispatch.loaded_as_persistent)
    if persistent_evt:
        loaded_as_persistent = context.session.dispatch.loaded_as_persistent
    instance_state = attributes.instance_state
    instance_dict = attributes.instance_dict
    session_id = context.session.hash_key
    runid = context.runid
    identity_token = context.identity_token
    version_check = context.version_check
    if version_check:
        version_id_col = mapper.version_id_col
        if version_id_col is not None:
            if adapter:
                version_id_col = adapter.columns[version_id_col]
            version_id_getter = result._getter(version_id_col)
        else:
            version_id_getter = None
    if not refresh_state and _polymorphic_from is not None:
        key = ('loader', path.path)
        if key in context.attributes and context.attributes[key].strategy == (('selectinload_polymorphic', True),):
            option_entities = context.attributes[key].local_opts['entities']
        else:
            option_entities = None
        selectin_load_via = mapper._should_selectin_load(option_entities, _polymorphic_from)
        if selectin_load_via and selectin_load_via is not _polymorphic_from:
            assert only_load_props is None
            callable_ = _load_subclass_via_in(context, path, selectin_load_via, _polymorphic_from, option_entities)
            PostLoad.callable_for_path(context, load_path, selectin_load_via.mapper, selectin_load_via, callable_, selectin_load_via)
    post_load = PostLoad.for_context(context, load_path, only_load_props)
    if refresh_state:
        refresh_identity_key = refresh_state.key
        if refresh_identity_key is None:
            refresh_identity_key = mapper._identity_key_from_state(refresh_state)
    else:
        refresh_identity_key = None
        primary_key_getter = getters['primary_key_getter']
    if mapper.allow_partial_pks:
        is_not_primary_key = _none_set.issuperset
    else:
        is_not_primary_key = _none_set.intersection

    def _instance(row):
        if refresh_identity_key:
            state = refresh_state
            instance = state.obj()
            dict_ = instance_dict(instance)
            isnew = state.runid != runid
            currentload = True
            loaded_instance = False
        else:
            identitykey = (identity_class, primary_key_getter(row), identity_token)
            instance = session_identity_map.get(identitykey)
            if instance is not None:
                state = instance_state(instance)
                dict_ = instance_dict(instance)
                isnew = state.runid != runid
                currentload = not isnew
                loaded_instance = False
                if version_check and version_id_getter and (not currentload):
                    _validate_version_id(mapper, state, dict_, row, version_id_getter)
            else:
                if is_not_primary_key(identitykey[1]):
                    return None
                isnew = True
                currentload = True
                loaded_instance = True
                instance = mapper.class_manager.new_instance()
                dict_ = instance_dict(instance)
                state = instance_state(instance)
                state.key = identitykey
                state.identity_token = identity_token
                state.session_id = session_id
                session_identity_map._add_unpresent(state, identitykey)
        effective_populate_existing = populate_existing
        if refresh_state is state:
            effective_populate_existing = True
        if currentload or effective_populate_existing:
            if isnew and (propagated_loader_options or not effective_populate_existing):
                state.load_options = propagated_loader_options
                state.load_path = load_path
            _populate_full(context, row, state, dict_, isnew, load_path, loaded_instance, effective_populate_existing, populators)
            if isnew:
                existing_runid = state.runid
                if loaded_instance:
                    if load_evt:
                        state.manager.dispatch.load(state, context)
                        if state.runid != existing_runid:
                            _warn_for_runid_changed(state)
                    if persistent_evt:
                        loaded_as_persistent(context.session, state)
                        if state.runid != existing_runid:
                            _warn_for_runid_changed(state)
                elif refresh_evt:
                    state.manager.dispatch.refresh(state, context, only_load_props)
                    if state.runid != runid:
                        _warn_for_runid_changed(state)
                if effective_populate_existing or state.modified:
                    if refresh_state and only_load_props:
                        state._commit(dict_, only_load_props)
                    else:
                        state._commit_all(dict_, session_identity_map)
            if post_load:
                post_load.add_state(state, True)
        else:
            unloaded = state.unloaded
            isnew = state not in context.partials
            if not isnew or unloaded or populators['eager']:
                to_load = _populate_partial(context, row, state, dict_, isnew, load_path, unloaded, populators)
                if isnew:
                    if refresh_evt:
                        existing_runid = state.runid
                        state.manager.dispatch.refresh(state, context, to_load)
                        if state.runid != existing_runid:
                            _warn_for_runid_changed(state)
                    state._commit(dict_, to_load)
            if post_load and context.invoke_all_eagers:
                post_load.add_state(state, False)
        return instance
    if mapper.polymorphic_map and (not _polymorphic_from) and (not refresh_state):

        def ensure_no_pk(row):
            identitykey = (identity_class, primary_key_getter(row), identity_token)
            if not is_not_primary_key(identitykey[1]):
                return identitykey
            else:
                return None
        _instance = _decorate_polymorphic_switch(_instance, context, query_entity, mapper, result, path, polymorphic_discriminator, adapter, ensure_no_pk)
    return _instance