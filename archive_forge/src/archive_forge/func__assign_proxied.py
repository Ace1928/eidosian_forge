from __future__ import annotations
import abc
import functools
from typing import Any
from typing import AsyncGenerator
from typing import AsyncIterator
from typing import Awaitable
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Generator
from typing import Generic
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeVar
import weakref
from . import exc as async_exc
from ... import util
from ...util.typing import Literal
from ...util.typing import Self
def _assign_proxied(self, target: Optional[_PT]) -> Optional[_PT]:
    if target is not None:
        target_ref: weakref.ref[_PT] = weakref.ref(target, ReversibleProxy._target_gced)
        proxy_ref = weakref.ref(self, functools.partial(ReversibleProxy._target_gced, target_ref))
        ReversibleProxy._proxy_objects[target_ref] = proxy_ref
    return target