from __future__ import annotations
import threading
import traceback
import typing
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref
from .base import _AsyncConnDialect
from .base import _ConnectionFairy
from .base import _ConnectionRecord
from .base import _CreatorFnType
from .base import _CreatorWRecFnType
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .. import exc
from .. import util
from ..util import chop_traceback
from ..util import queue as sqla_queue
from ..util.typing import Literal
def _transfer_from(self, other_static_pool: StaticPool) -> None:

    def creator(rec: ConnectionPoolEntry) -> DBAPIConnection:
        conn = other_static_pool.connection.dbapi_connection
        assert conn is not None
        return conn
    self._invoke_creator = creator