import asyncio
import sys
from asyncio import AbstractEventLoop, AbstractEventLoopPolicy
from contextlib import suppress
from typing import Any, Callable, Dict, Optional, Sequence, Type
from warnings import catch_warnings, filterwarnings, warn
from twisted.internet import asyncioreactor, error
from twisted.internet.base import DelayedCall
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.misc import load_object
class CallLaterOnce:
    """Schedule a function to be called in the next reactor loop, but only if
    it hasn't been already scheduled since the last time it ran.
    """

    def __init__(self, func: Callable, *a: Any, **kw: Any):
        self._func: Callable = func
        self._a: Sequence[Any] = a
        self._kw: Dict[str, Any] = kw
        self._call: Optional[DelayedCall] = None

    def schedule(self, delay: float=0) -> None:
        from twisted.internet import reactor
        if self._call is None:
            self._call = reactor.callLater(delay, self)

    def cancel(self) -> None:
        if self._call:
            self._call.cancel()

    def __call__(self) -> Any:
        self._call = None
        return self._func(*self._a, **self._kw)