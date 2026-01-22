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
def _load_via_parent(self, our_states, query_info, q, context, execution_options):
    uselist = self.uselist
    _empty_result = () if uselist else None
    while our_states:
        chunk = our_states[0:self._chunksize]
        our_states = our_states[self._chunksize:]
        primary_keys = [key[0] if query_info.zero_idx else key for key, state, state_dict, overwrite in chunk]
        data = collections.defaultdict(list)
        for k, v in itertools.groupby(context.session.execute(q, params={'primary_keys': primary_keys}, execution_options=execution_options).unique(), lambda x: x[0]):
            data[k].extend((vv[1] for vv in v))
        for key, state, state_dict, overwrite in chunk:
            if not overwrite and self.key in state_dict:
                continue
            collection = data.get(key, _empty_result)
            if not uselist and collection:
                if len(collection) > 1:
                    util.warn("Multiple rows returned with uselist=False for eagerly-loaded attribute '%s' " % self)
                state.get_impl(self.key).set_committed_value(state, state_dict, collection[0])
            else:
                state.get_impl(self.key).set_committed_value(state, state_dict, collection)