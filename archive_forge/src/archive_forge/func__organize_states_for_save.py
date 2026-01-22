from __future__ import annotations
from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator
from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
def _organize_states_for_save(base_mapper, states, uowtransaction):
    """Make an initial pass across a set of states for INSERT or
    UPDATE.

    This includes splitting out into distinct lists for
    each, calling before_insert/before_update, obtaining
    key information for each state including its dictionary,
    mapper, the connection to use for the execution per state,
    and the identity flag.

    """
    for state, dict_, mapper, connection in _connections_for_states(base_mapper, uowtransaction, states):
        has_identity = bool(state.key)
        instance_key = state.key or mapper._identity_key_from_state(state)
        row_switch = update_version_id = None
        if not has_identity:
            mapper.dispatch.before_insert(mapper, connection, state)
        else:
            mapper.dispatch.before_update(mapper, connection, state)
        if mapper._validate_polymorphic_identity:
            mapper._validate_polymorphic_identity(mapper, state, dict_)
        if not has_identity and instance_key in uowtransaction.session.identity_map:
            instance = uowtransaction.session.identity_map[instance_key]
            existing = attributes.instance_state(instance)
            if not uowtransaction.was_already_deleted(existing):
                if not uowtransaction.is_deleted(existing):
                    util.warn('New instance %s with identity key %s conflicts with persistent instance %s' % (state_str(state), instance_key, state_str(existing)))
                else:
                    base_mapper._log_debug('detected row switch for identity %s.  will update %s, remove %s from transaction', instance_key, state_str(state), state_str(existing))
                    uowtransaction.remove_state_actions(existing)
                    row_switch = existing
        if (has_identity or row_switch) and mapper.version_id_col is not None:
            update_version_id = mapper._get_committed_state_attr_by_column(row_switch if row_switch else state, row_switch.dict if row_switch else dict_, mapper.version_id_col)
        yield (state, dict_, mapper, connection, has_identity, row_switch, update_version_id)