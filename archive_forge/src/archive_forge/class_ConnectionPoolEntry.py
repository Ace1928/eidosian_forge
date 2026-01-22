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
class ConnectionPoolEntry(ManagesConnection):
    """Interface for the object that maintains an individual database
    connection on behalf of a :class:`_pool.Pool` instance.

    The :class:`.ConnectionPoolEntry` object represents the long term
    maintainance of a particular connection for a pool, including expiring or
    invalidating that connection to have it replaced with a new one, which will
    continue to be maintained by that same :class:`.ConnectionPoolEntry`
    instance. Compared to :class:`.PoolProxiedConnection`, which is the
    short-term, per-checkout connection manager, this object lasts for the
    lifespan of a particular "slot" within a connection pool.

    The :class:`.ConnectionPoolEntry` object is mostly visible to public-facing
    API code when it is delivered to connection pool event hooks, such as
    :meth:`_events.PoolEvents.connect` and :meth:`_events.PoolEvents.checkout`.

    .. versionadded:: 2.0  :class:`.ConnectionPoolEntry` provides the public
       facing interface for the :class:`._ConnectionRecord` internal class.

    """
    __slots__ = ()

    @property
    def in_use(self) -> bool:
        """Return True the connection is currently checked out"""
        raise NotImplementedError()

    def close(self) -> None:
        """Close the DBAPI connection managed by this connection pool entry."""
        raise NotImplementedError()