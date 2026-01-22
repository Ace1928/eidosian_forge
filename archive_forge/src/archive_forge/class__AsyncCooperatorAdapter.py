import asyncio
import inspect
from asyncio import Future
from functools import wraps
from types import CoroutineType
from typing import (
from twisted.internet import defer
from twisted.internet.defer import Deferred, DeferredList, ensureDeferred
from twisted.internet.task import Cooperator
from twisted.python import failure
from twisted.python.failure import Failure
from scrapy.exceptions import IgnoreRequest
from scrapy.utils.reactor import _get_asyncio_event_loop, is_asyncio_reactor_installed
class _AsyncCooperatorAdapter(Iterator):
    """A class that wraps an async iterable into a normal iterator suitable
    for using in Cooperator.coiterate(). As it's only needed for parallel_async(),
    it calls the callable directly in the callback, instead of providing a more
    generic interface.

    On the outside, this class behaves as an iterator that yields Deferreds.
    Each Deferred is fired with the result of the callable which was called on
    the next result from aiterator. It raises StopIteration when aiterator is
    exhausted, as expected.

    Cooperator calls __next__() multiple times and waits on the Deferreds
    returned from it. As async generators (since Python 3.8) don't support
    awaiting on __anext__() several times in parallel, we need to serialize
    this. It's done by storing the Deferreds returned from __next__() and
    firing the oldest one when a result from __anext__() is available.

    The workflow:
    1. When __next__() is called for the first time, it creates a Deferred, stores it
    in self.waiting_deferreds and returns it. It also makes a Deferred that will wait
    for self.aiterator.__anext__() and puts it into self.anext_deferred.
    2. If __next__() is called again before self.anext_deferred fires, more Deferreds
    are added to self.waiting_deferreds.
    3. When self.anext_deferred fires, it either calls _callback() or _errback(). Both
    clear self.anext_deferred.
    3.1. _callback() calls the callable passing the result value that it takes, pops a
    Deferred from self.waiting_deferreds, and if the callable result was a Deferred, it
    chains those Deferreds so that the waiting Deferred will fire when the result
    Deferred does, otherwise it fires it directly. This causes one awaiting task to
    receive a result. If self.waiting_deferreds is still not empty, new __anext__() is
    called and self.anext_deferred is populated.
    3.2. _errback() checks the exception class. If it's StopAsyncIteration it means
    self.aiterator is exhausted and so it sets self.finished and fires all
    self.waiting_deferreds. Other exceptions are propagated.
    4. If __next__() is called after __anext__() was handled, then if self.finished is
    True, it raises StopIteration, otherwise it acts like in step 2, but if
    self.anext_deferred is now empty is also populates it with a new __anext__().

    Note that CooperativeTask ignores the value returned from the Deferred that it waits
    for, so we fire them with None when needed.

    It may be possible to write an async iterator-aware replacement for
    Cooperator/CooperativeTask and use it instead of this adapter to achieve the same
    goal.
    """

    def __init__(self, aiterable: AsyncIterable, callable: Callable, *callable_args: Any, **callable_kwargs: Any):
        self.aiterator: AsyncIterator = aiterable.__aiter__()
        self.callable: Callable = callable
        self.callable_args: Tuple[Any, ...] = callable_args
        self.callable_kwargs: Dict[str, Any] = callable_kwargs
        self.finished: bool = False
        self.waiting_deferreds: List[Deferred] = []
        self.anext_deferred: Optional[Deferred] = None

    def _callback(self, result: Any) -> None:
        self.anext_deferred = None
        result = self.callable(result, *self.callable_args, **self.callable_kwargs)
        d = self.waiting_deferreds.pop(0)
        if isinstance(result, Deferred):
            result.chainDeferred(d)
        else:
            d.callback(None)
        if self.waiting_deferreds:
            self._call_anext()

    def _errback(self, failure: Failure) -> None:
        self.anext_deferred = None
        failure.trap(StopAsyncIteration)
        self.finished = True
        for d in self.waiting_deferreds:
            d.callback(None)

    def _call_anext(self) -> None:
        self.anext_deferred = deferred_from_coro(self.aiterator.__anext__())
        self.anext_deferred.addCallbacks(self._callback, self._errback)

    def __next__(self) -> Deferred:
        if self.finished:
            raise StopIteration
        d: Deferred = Deferred()
        self.waiting_deferreds.append(d)
        if not self.anext_deferred:
            self._call_anext()
        return d