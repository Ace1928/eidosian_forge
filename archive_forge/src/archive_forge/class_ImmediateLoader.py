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
@relationships.RelationshipProperty.strategy_for(lazy='immediate')
class ImmediateLoader(PostLoader):
    __slots__ = ('join_depth',)

    def __init__(self, parent, strategy_key):
        super().__init__(parent, strategy_key)
        self.join_depth = self.parent_property.join_depth

    def init_class_attribute(self, mapper):
        self.parent_property._get_strategy((('lazy', 'select'),)).init_class_attribute(mapper)

    def create_row_processor(self, context, query_entity, path, loadopt, mapper, result, adapter, populators):
        effective_path, run_loader, execution_options, recursion_depth = self._setup_for_recursion(context, path, loadopt, self.join_depth)
        if not run_loader:
            flags = attributes.PASSIVE_NO_FETCH_RELATED | PassiveFlag.NO_RAISE
        else:
            flags = attributes.PASSIVE_OFF | PassiveFlag.NO_RAISE
        loading.PostLoad.callable_for_path(context, effective_path, self.parent, self.parent_property, self._load_for_path, loadopt, flags, recursion_depth, execution_options)

    def _load_for_path(self, context, path, states, load_only, loadopt, flags, recursion_depth, execution_options):
        if recursion_depth:
            new_opt = Load(loadopt.path.entity)
            new_opt.context = (loadopt, loadopt._recurse())
            alternate_effective_path = path._truncate_recursive()
            extra_options = (new_opt,)
        else:
            new_opt = None
            alternate_effective_path = path
            extra_options = ()
        key = self.key
        lazyloader = self.parent_property._get_strategy((('lazy', 'select'),))
        for state, overwrite in states:
            dict_ = state.dict
            if overwrite or key not in dict_:
                value = lazyloader._load_for_state(state, flags, extra_options=extra_options, alternate_effective_path=alternate_effective_path, execution_options=execution_options)
                if value not in (ATTR_WAS_SET, LoaderCallableStatus.PASSIVE_NO_RESULT):
                    state.get_impl(key).set_committed_value(state, dict_, value)