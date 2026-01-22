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
class _ConnDialect:
    """partial implementation of :class:`.Dialect`
    which provides DBAPI connection methods.

    When a :class:`_pool.Pool` is combined with an :class:`_engine.Engine`,
    the :class:`_engine.Engine` replaces this with its own
    :class:`.Dialect`.

    """
    is_async = False
    has_terminate = False

    def do_rollback(self, dbapi_connection: PoolProxiedConnection) -> None:
        dbapi_connection.rollback()

    def do_commit(self, dbapi_connection: PoolProxiedConnection) -> None:
        dbapi_connection.commit()

    def do_terminate(self, dbapi_connection: DBAPIConnection) -> None:
        dbapi_connection.close()

    def do_close(self, dbapi_connection: DBAPIConnection) -> None:
        dbapi_connection.close()

    def _do_ping_w_event(self, dbapi_connection: DBAPIConnection) -> bool:
        raise NotImplementedError('The ping feature requires that a dialect is passed to the connection pool.')

    def get_driver_connection(self, connection: DBAPIConnection) -> Any:
        return connection