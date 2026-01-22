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
def _connections_for_states(base_mapper, uowtransaction, states):
    """Return an iterator of (state, state.dict, mapper, connection).

    The states are sorted according to _sort_states, then paired
    with the connection they should be using for the given
    unit of work transaction.

    """
    if uowtransaction.session.connection_callable:
        connection_callable = uowtransaction.session.connection_callable
    else:
        connection = uowtransaction.transaction.connection(base_mapper)
        connection_callable = None
    for state in _sort_states(base_mapper, states):
        if connection_callable:
            connection = connection_callable(base_mapper, state.obj())
        mapper = state.manager.mapper
        yield (state, state.dict, mapper, connection)