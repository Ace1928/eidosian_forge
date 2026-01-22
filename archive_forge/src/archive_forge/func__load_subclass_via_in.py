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
def _load_subclass_via_in(context, path, entity, polymorphic_from, option_entities):
    mapper = entity.mapper
    polymorphic_from_mapper = polymorphic_from.mapper
    not_against_basemost = polymorphic_from_mapper.inherits is not None
    zero_idx = len(mapper.base_mapper.primary_key) == 1
    if entity.is_aliased_class or not_against_basemost:
        q, enable_opt, disable_opt = mapper._subclass_load_via_in(entity, polymorphic_from)
    else:
        q, enable_opt, disable_opt = mapper._subclass_load_via_in_mapper

    def do_load(context, path, states, load_only, effective_entity):
        if not option_entities:
            states = [(s, v) for s, v in states if s.mapper._would_selectin_load_only_from_given_mapper(mapper)]
            if not states:
                return
        orig_query = context.query
        if path.parent:
            enable_opt_lcl = enable_opt._prepend_path(path)
            disable_opt_lcl = disable_opt._prepend_path(path)
        else:
            enable_opt_lcl = enable_opt
            disable_opt_lcl = disable_opt
        options = (enable_opt_lcl,) + orig_query._with_options + (disable_opt_lcl,)
        q2 = q.options(*options)
        q2._compile_options = context.compile_state.default_compile_options
        q2._compile_options += {'_current_path': path.parent}
        if context.populate_existing:
            q2 = q2.execution_options(populate_existing=True)
        context.session.execute(q2, dict(primary_keys=[state.key[1][0] if zero_idx else state.key[1] for state, load_attrs in states])).unique().scalars().all()
    return do_load