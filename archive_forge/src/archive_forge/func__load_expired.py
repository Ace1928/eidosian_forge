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
def _load_expired(self, state: InstanceState[_O], passive: PassiveFlag) -> LoaderCallableStatus:
    """__call__ allows the InstanceState to act as a deferred
        callable for loading expired attributes, which is also
        serializable (picklable).

        """
    if not passive & SQL_OK:
        return PASSIVE_NO_RESULT
    toload = self.expired_attributes.intersection(self.unmodified)
    toload = toload.difference((attr for attr in toload if not self.manager[attr].impl.load_on_unexpire))
    self.manager.expired_attribute_loader(self, toload, passive)
    self.expired_attributes.clear()
    return ATTR_WAS_SET