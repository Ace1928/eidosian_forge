from __future__ import annotations
import contextlib
from enum import Enum
import itertools
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref
from . import attributes
from . import bulk_persistence
from . import context
from . import descriptor_props
from . import exc
from . import identity
from . import loading
from . import query
from . import state as statelib
from ._typing import _O
from ._typing import insp_is_mapper
from ._typing import is_composite_class
from ._typing import is_orm_option
from ._typing import is_user_defined_option
from .base import _class_to_mapper
from .base import _none_set
from .base import _state_mapper
from .base import instance_str
from .base import LoaderCallableStatus
from .base import object_mapper
from .base import object_state
from .base import PassiveFlag
from .base import state_str
from .context import FromStatement
from .context import ORMCompileState
from .identity import IdentityMap
from .query import Query
from .state import InstanceState
from .state_changes import _StateChange
from .state_changes import _StateChangeState
from .state_changes import _StateChangeStates
from .unitofwork import UOWTransaction
from .. import engine
from .. import exc as sa_exc
from .. import sql
from .. import util
from ..engine import Connection
from ..engine import Engine
from ..engine.util import TransactionalContext
from ..event import dispatcher
from ..event import EventTarget
from ..inspection import inspect
from ..inspection import Inspectable
from ..sql import coercions
from ..sql import dml
from ..sql import roles
from ..sql import Select
from ..sql import TableClause
from ..sql import visitors
from ..sql.base import _NoArg
from ..sql.base import CompileState
from ..sql.schema import Table
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import IdentitySet
from ..util.typing import Literal
from ..util.typing import Protocol
def _register_persistent(self, states: Set[InstanceState[Any]]) -> None:
    """Register all persistent objects from a flush.

        This is used both for pending objects moving to the persistent
        state as well as already persistent objects.

        """
    pending_to_persistent = self.dispatch.pending_to_persistent or None
    for state in states:
        mapper = _state_mapper(state)
        obj = state.obj()
        if obj is not None:
            instance_key = mapper._identity_key_from_state(state)
            if _none_set.intersection(instance_key[1]) and (not mapper.allow_partial_pks) or _none_set.issuperset(instance_key[1]):
                raise exc.FlushError('Instance %s has a NULL identity key.  If this is an auto-generated value, check that the database table allows generation of new primary key values, and that the mapped Column object is configured to expect these generated values.  Ensure also that this flush() is not occurring at an inappropriate time, such as within a load() event.' % state_str(state))
            if state.key is None:
                state.key = instance_key
            elif state.key != instance_key:
                self.identity_map.safe_discard(state)
                trans = self._transaction
                assert trans is not None
                if state in trans._key_switches:
                    orig_key = trans._key_switches[state][0]
                else:
                    orig_key = state.key
                trans._key_switches[state] = (orig_key, instance_key)
                state.key = instance_key
            old = self.identity_map.replace(state)
            if old is not None and mapper._identity_key_from_state(old) == instance_key and (old.obj() is not None):
                util.warn('Identity map already had an identity for %s, replacing it with newly flushed object.   Are there load operations occurring inside of an event handler within the flush?' % (instance_key,))
            state._orphaned_outside_of_session = False
    statelib.InstanceState._commit_all_states(((state, state.dict) for state in states), self.identity_map)
    self._register_altered(states)
    if pending_to_persistent is not None:
        for state in states.intersection(self._new):
            pending_to_persistent(self, state)
    for state in set(states).intersection(self._new):
        self._new.pop(state)