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
def _expire_attributes(self, dict_: _InstanceDict, attribute_names: Iterable[str], no_loader: bool=False) -> None:
    pending = self.__dict__.get('_pending_mutations', None)
    callables = self.callables
    for key in attribute_names:
        impl = self.manager[key].impl
        if impl.accepts_scalar_loader:
            if no_loader and (impl.callable_ or key in callables):
                continue
            self.expired_attributes.add(key)
            if callables and key in callables:
                del callables[key]
        old = dict_.pop(key, NO_VALUE)
        if is_collection_impl(impl) and old is not NO_VALUE:
            impl._invalidate_collection(old)
        lkv = self._last_known_values
        if lkv is not None and key in lkv and (old is not NO_VALUE):
            lkv[key] = old
        self.committed_state.pop(key, None)
        if pending:
            pending.pop(key, None)
    self.manager.dispatch.expire(self, attribute_names)