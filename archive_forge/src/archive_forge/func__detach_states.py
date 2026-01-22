from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from . import base
from . import exc as orm_exc
from . import interfaces
from ._typing import _O
from ._typing import is_collection_impl
from .base import ATTR_WAS_SET
from .base import INIT_OK
from .base import LoaderCallableStatus
from .base import NEVER_SET
from .base import NO_VALUE
from .base import PASSIVE_NO_INITIALIZE
from .base import PASSIVE_NO_RESULT
from .base import PASSIVE_OFF
from .base import SQL_OK
from .path_registry import PathRegistry
from .. import exc as sa_exc
from .. import inspection
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
@classmethod
def _detach_states(self, states: Iterable[InstanceState[_O]], session: Session, to_transient: bool=False) -> None:
    persistent_to_detached = session.dispatch.persistent_to_detached or None
    deleted_to_detached = session.dispatch.deleted_to_detached or None
    pending_to_transient = session.dispatch.pending_to_transient or None
    persistent_to_transient = session.dispatch.persistent_to_transient or None
    for state in states:
        deleted = state._deleted
        pending = state.key is None
        persistent = not pending and (not deleted)
        state.session_id = None
        if to_transient and state.key:
            del state.key
        if persistent:
            if to_transient:
                if persistent_to_transient is not None:
                    persistent_to_transient(session, state)
            elif persistent_to_detached is not None:
                persistent_to_detached(session, state)
        elif deleted and deleted_to_detached is not None:
            deleted_to_detached(session, state)
        elif pending and pending_to_transient is not None:
            pending_to_transient(session, state)
        state._strong_obj = None