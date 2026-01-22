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
def __connect(self) -> None:
    pool = self.__pool
    self.dbapi_connection = None
    try:
        self.starttime = time.time()
        self.dbapi_connection = connection = pool._invoke_creator(self)
        pool.logger.debug('Created new connection %r', connection)
        self.fresh = True
    except BaseException as e:
        with util.safe_reraise():
            pool.logger.debug('Error on connect(): %s', e)
    else:
        if pool.dispatch.first_connect:
            pool.dispatch.first_connect.for_modify(pool.dispatch).exec_once_unless_exception(self.dbapi_connection, self)
        pool.dispatch.connect.for_modify(pool.dispatch)._exec_w_sync_on_first_run(self.dbapi_connection, self)