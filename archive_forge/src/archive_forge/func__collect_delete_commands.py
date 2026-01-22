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
def _collect_delete_commands(base_mapper, uowtransaction, table, states_to_delete):
    """Identify values to use in DELETE statements for a list of
    states to be deleted."""
    for state, state_dict, mapper, connection, update_version_id in states_to_delete:
        if table not in mapper._pks_by_table:
            continue
        params = {}
        for col in mapper._pks_by_table[table]:
            params[col.key] = value = mapper._get_committed_state_attr_by_column(state, state_dict, col)
            if value is None:
                raise orm_exc.FlushError("Can't delete from table %s using NULL for primary key value on column %s" % (table, col))
        if update_version_id is not None and mapper.version_id_col in mapper._cols_by_table[table]:
            params[mapper.version_id_col.key] = update_version_id
        yield (params, connection)