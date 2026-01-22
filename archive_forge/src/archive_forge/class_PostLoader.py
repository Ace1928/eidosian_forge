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
class PostLoader(AbstractRelationshipLoader):
    """A relationship loader that emits a second SELECT statement."""
    __slots__ = ()

    def _setup_for_recursion(self, context, path, loadopt, join_depth=None):
        effective_path = (context.compile_state.current_path or orm_util.PathRegistry.root) + path
        top_level_context = context._get_top_level_context()
        execution_options = util.immutabledict({'sa_top_level_orm_context': top_level_context})
        if loadopt:
            recursion_depth = loadopt.local_opts.get('recursion_depth', None)
            unlimited_recursion = recursion_depth == -1
        else:
            recursion_depth = None
            unlimited_recursion = False
        if recursion_depth is not None:
            if not self.parent_property._is_self_referential:
                raise sa_exc.InvalidRequestError(f'recursion_depth option on relationship {self.parent_property} not valid for non-self-referential relationship')
            recursion_depth = context.execution_options.get(f'_recursion_depth_{id(self)}', recursion_depth)
            if not unlimited_recursion and recursion_depth < 0:
                return (effective_path, False, execution_options, recursion_depth)
            if not unlimited_recursion:
                execution_options = execution_options.union({f'_recursion_depth_{id(self)}': recursion_depth - 1})
        if loading.PostLoad.path_exists(context, effective_path, self.parent_property):
            return (effective_path, False, execution_options, recursion_depth)
        path_w_prop = path[self.parent_property]
        effective_path_w_prop = effective_path[self.parent_property]
        if not path_w_prop.contains(context.attributes, 'loader'):
            if join_depth:
                if effective_path_w_prop.length / 2 > join_depth:
                    return (effective_path, False, execution_options, recursion_depth)
            elif effective_path_w_prop.contains_mapper(self.mapper):
                return (effective_path, False, execution_options, recursion_depth)
        return (effective_path, True, execution_options, recursion_depth)