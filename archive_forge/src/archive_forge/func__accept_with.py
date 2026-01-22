from __future__ import annotations
import typing
from typing import Any
from typing import Optional
from typing import Type
from typing import Union
from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .base import PoolResetState
from .. import event
from .. import util
@util.preload_module('sqlalchemy.engine')
@classmethod
def _accept_with(cls, target: Union[Pool, Type[Pool], Engine, Type[Engine]], identifier: str) -> Optional[Union[Pool, Type[Pool]]]:
    if not typing.TYPE_CHECKING:
        Engine = util.preloaded.engine.Engine
    if isinstance(target, type):
        if issubclass(target, Engine):
            return Pool
        else:
            assert issubclass(target, Pool)
            return target
    elif isinstance(target, Engine):
        return target.pool
    elif isinstance(target, Pool):
        return target
    elif hasattr(target, '_no_async_engine_events'):
        target._no_async_engine_events()
    else:
        return None