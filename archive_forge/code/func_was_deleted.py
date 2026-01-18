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
@property
def was_deleted(self) -> bool:
    """Return True if this object is or was previously in the
        "deleted" state and has not been reverted to persistent.

        This flag returns True once the object was deleted in flush.
        When the object is expunged from the session either explicitly
        or via transaction commit and enters the "detached" state,
        this flag will continue to report True.

        .. seealso::

            :attr:`.InstanceState.deleted` - refers to the "deleted" state

            :func:`.orm.util.was_deleted` - standalone function

            :ref:`session_object_states`

        """
    return self._deleted