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
class ProxyComparable(ReversibleProxy[_PT]):
    __slots__ = ()

    @util.ro_non_memoized_property
    def _proxied(self) -> _PT:
        raise NotImplementedError()

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._proxied == other._proxied

    def __ne__(self, other: Any) -> bool:
        return not isinstance(other, self.__class__) or self._proxied != other._proxied