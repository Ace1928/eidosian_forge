import sys
import asyncio
import functools
from typing import Optional, Any, Callable, Awaitable, Union, TypeVar, Coroutine, Iterable, AsyncIterable, AsyncIterator, AsyncGenerator
from lazyops.utils.system import get_cpu_count
from lazyops.utils.pooler import ThreadPooler
from lazyops.types.common import UpperStrEnum
class ReturnWhenType(UpperStrEnum):
    """
    Return When Type
    """
    FIRST_COMPLETED = 'FIRST_COMPLETED'
    FIRST_EXCEPTION = 'FIRST_EXCEPTION'
    ALL_COMPLETED = 'ALL_COMPLETED'

    @property
    def val(self) -> Union[asyncio.FIRST_COMPLETED, asyncio.FIRST_EXCEPTION, asyncio.ALL_COMPLETED]:
        """
        Get the value of the return when type
        """
        return getattr(asyncio, self.value)