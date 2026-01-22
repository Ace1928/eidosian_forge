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
def _modified_event(self, dict_: _InstanceDict, attr: Optional[AttributeImpl], previous: Any, collection: bool=False, is_userland: bool=False) -> None:
    if attr:
        if not attr.send_modified_events:
            return
        if is_userland and attr.key not in dict_:
            raise sa_exc.InvalidRequestError("Can't flag attribute '%s' modified; it's not present in the object state" % attr.key)
        if attr.key not in self.committed_state or is_userland:
            if collection:
                if TYPE_CHECKING:
                    assert is_collection_impl(attr)
                if previous is NEVER_SET:
                    if attr.key in dict_:
                        previous = dict_[attr.key]
                if previous not in (None, NO_VALUE, NEVER_SET):
                    previous = attr.copy(previous)
            self.committed_state[attr.key] = previous
        lkv = self._last_known_values
        if lkv is not None and attr.key in lkv:
            lkv[attr.key] = NO_VALUE
    if self.session_id and self._strong_obj is None or not self.modified:
        self.modified = True
        instance_dict = self._instance_dict()
        if instance_dict:
            has_modified = bool(instance_dict._modified)
            instance_dict._modified.add(self)
        else:
            has_modified = False
        inst = self.obj()
        if self.session_id:
            self._strong_obj = inst
            if not has_modified:
                try:
                    session = _sessions[self.session_id]
                except KeyError:
                    pass
                else:
                    if session._transaction is None:
                        session._autobegin_t()
        if inst is None and attr:
            raise orm_exc.ObjectDereferencedError("Can't emit change event for attribute '%s' - parent object of type %s has been garbage collected." % (self.manager[attr.key], base.state_class_str(self)))