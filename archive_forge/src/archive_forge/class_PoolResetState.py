from __future__ import annotations
from collections import deque
import dataclasses
from enum import Enum
import threading
import time
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Deque
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .. import event
from .. import exc
from .. import log
from .. import util
from ..util.typing import Literal
from ..util.typing import Protocol
@dataclasses.dataclass(frozen=True)
class PoolResetState:
    """describes the state of a DBAPI connection as it is being passed to
    the :meth:`.PoolEvents.reset` connection pool event.

    .. versionadded:: 2.0.0b3

    """
    __slots__ = ('transaction_was_reset', 'terminate_only', 'asyncio_safe')
    transaction_was_reset: bool
    'Indicates if the transaction on the DBAPI connection was already\n    essentially "reset" back by the :class:`.Connection` object.\n\n    This boolean is True if the :class:`.Connection` had transactional\n    state present upon it, which was then not closed using the\n    :meth:`.Connection.rollback` or :meth:`.Connection.commit` method;\n    instead, the transaction was closed inline within the\n    :meth:`.Connection.close` method so is guaranteed to remain non-present\n    when this event is reached.\n\n    '
    terminate_only: bool
    "indicates if the connection is to be immediately terminated and\n    not checked in to the pool.\n\n    This occurs for connections that were invalidated, as well as asyncio\n    connections that were not cleanly handled by the calling code that\n    are instead being garbage collected.   In the latter case,\n    operations can't be safely run on asyncio connections within garbage\n    collection as there is not necessarily an event loop present.\n\n    "
    asyncio_safe: bool
    'Indicates if the reset operation is occurring within a scope where\n    an enclosing event loop is expected to be present for asyncio applications.\n\n    Will be False in the case that the connection is being garbage collected.\n\n    '