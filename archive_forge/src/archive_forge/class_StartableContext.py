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
class StartableContext(Awaitable[_T_co], abc.ABC):
    __slots__ = ()

    @abc.abstractmethod
    async def start(self, is_ctxmanager: bool=False) -> _T_co:
        raise NotImplementedError()

    def __await__(self) -> Generator[Any, Any, _T_co]:
        return self.start().__await__()

    async def __aenter__(self) -> _T_co:
        return await self.start(is_ctxmanager=True)

    @abc.abstractmethod
    async def __aexit__(self, type_: Any, value: Any, traceback: Any) -> Optional[bool]:
        pass

    def _raise_for_not_started(self) -> NoReturn:
        raise async_exc.AsyncContextNotStarted('%s context has not been started and object has not been awaited.' % self.__class__.__name__)