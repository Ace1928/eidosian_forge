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
def _register_attribute(prop, mapper, useobject, compare_function=None, typecallable=None, callable_=None, proxy_property=None, active_history=False, impl_class=None, **kw):
    listen_hooks = []
    uselist = useobject and prop.uselist
    if useobject and prop.single_parent:
        listen_hooks.append(single_parent_validator)
    if prop.key in prop.parent.validators:
        fn, opts = prop.parent.validators[prop.key]
        listen_hooks.append(lambda desc, prop: orm_util._validator_events(desc, prop.key, fn, **opts))
    if useobject:
        listen_hooks.append(unitofwork.track_cascade_events)
    if useobject:
        backref = prop.back_populates
        if backref and prop._effective_sync_backref:
            listen_hooks.append(lambda desc, prop: attributes.backref_listeners(desc, backref, uselist))
    for m in mapper.self_and_descendants:
        if prop is m._props.get(prop.key) and (not m.class_manager._attr_has_impl(prop.key)):
            desc = attributes.register_attribute_impl(m.class_, prop.key, parent_token=prop, uselist=uselist, compare_function=compare_function, useobject=useobject, trackparent=useobject and (prop.single_parent or prop.direction is interfaces.ONETOMANY), typecallable=typecallable, callable_=callable_, active_history=active_history, impl_class=impl_class, send_modified_events=not useobject or not prop.viewonly, doc=prop.doc, **kw)
            for hook in listen_hooks:
                hook(desc, prop)