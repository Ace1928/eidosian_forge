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
class GeneratorStartableContext(StartableContext[_T_co]):
    __slots__ = ('gen',)
    gen: AsyncGenerator[_T_co, Any]

    def __init__(self, func: Callable[..., AsyncIterator[_T_co]], args: Tuple[Any, ...], kwds: Dict[str, Any]):
        self.gen = func(*args, **kwds)

    async def start(self, is_ctxmanager: bool=False) -> _T_co:
        try:
            start_value = await util.anext_(self.gen)
        except StopAsyncIteration:
            raise RuntimeError("generator didn't yield") from None
        if not is_ctxmanager:
            await self.gen.aclose()
        return start_value

    async def __aexit__(self, typ: Any, value: Any, traceback: Any) -> Optional[bool]:
        if typ is None:
            try:
                await util.anext_(self.gen)
            except StopAsyncIteration:
                return False
            else:
                raise RuntimeError("generator didn't stop")
        else:
            if value is None:
                value = typ()
            try:
                await self.gen.athrow(value)
            except StopAsyncIteration as exc:
                return exc is not value
            except RuntimeError as exc:
                if exc is value:
                    return False
                if isinstance(value, (StopIteration, StopAsyncIteration)) and exc.__cause__ is value:
                    return False
                raise
            except BaseException as exc:
                if exc is not value:
                    raise
                return False
            raise RuntimeError("generator didn't stop after athrow()")