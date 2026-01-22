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
def _load_via_child(self, our_states, none_states, query_info, q, context, execution_options):
    uselist = self.uselist
    our_keys = sorted(our_states)
    while our_keys:
        chunk = our_keys[0:self._chunksize]
        our_keys = our_keys[self._chunksize:]
        data = {k: v for k, v in context.session.execute(q, params={'primary_keys': [key[0] if query_info.zero_idx else key for key in chunk]}, execution_options=execution_options).unique()}
        for key in chunk:
            related_obj = data.get(key, None)
            for state, dict_, overwrite in our_states[key]:
                if not overwrite and self.key in dict_:
                    continue
                state.get_impl(self.key).set_committed_value(state, dict_, related_obj if not uselist else [related_obj])
    for state, dict_, overwrite in none_states:
        if not overwrite and self.key in dict_:
            continue
        state.get_impl(self.key).set_committed_value(state, dict_, None)