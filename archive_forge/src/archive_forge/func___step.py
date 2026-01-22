import concurrent.futures
import contextvars
import functools
import inspect
import itertools
import types
import warnings
import weakref
from types import GenericAlias
from . import base_tasks
from . import coroutines
from . import events
from . import exceptions
from . import futures
from .coroutines import _is_coroutine
def __step(self, exc=None):
    if self.done():
        raise exceptions.InvalidStateError(f'_step(): already done: {self!r}, {exc!r}')
    if self._must_cancel:
        if not isinstance(exc, exceptions.CancelledError):
            exc = self._make_cancelled_error()
        self._must_cancel = False
    coro = self._coro
    self._fut_waiter = None
    _enter_task(self._loop, self)
    try:
        if exc is None:
            result = coro.send(None)
        else:
            result = coro.throw(exc)
    except StopIteration as exc:
        if self._must_cancel:
            self._must_cancel = False
            super().cancel(msg=self._cancel_message)
        else:
            super().set_result(exc.value)
    except exceptions.CancelledError as exc:
        self._cancelled_exc = exc
        super().cancel()
    except (KeyboardInterrupt, SystemExit) as exc:
        super().set_exception(exc)
        raise
    except BaseException as exc:
        super().set_exception(exc)
    else:
        blocking = getattr(result, '_asyncio_future_blocking', None)
        if blocking is not None:
            if futures._get_loop(result) is not self._loop:
                new_exc = RuntimeError(f'Task {self!r} got Future {result!r} attached to a different loop')
                self._loop.call_soon(self.__step, new_exc, context=self._context)
            elif blocking:
                if result is self:
                    new_exc = RuntimeError(f'Task cannot await on itself: {self!r}')
                    self._loop.call_soon(self.__step, new_exc, context=self._context)
                else:
                    result._asyncio_future_blocking = False
                    result.add_done_callback(self.__wakeup, context=self._context)
                    self._fut_waiter = result
                    if self._must_cancel:
                        if self._fut_waiter.cancel(msg=self._cancel_message):
                            self._must_cancel = False
            else:
                new_exc = RuntimeError(f'yield was used instead of yield from in task {self!r} with {result!r}')
                self._loop.call_soon(self.__step, new_exc, context=self._context)
        elif result is None:
            self._loop.call_soon(self.__step, context=self._context)
        elif inspect.isgenerator(result):
            new_exc = RuntimeError(f'yield was used instead of yield from for generator in task {self!r} with {result!r}')
            self._loop.call_soon(self.__step, new_exc, context=self._context)
        else:
            new_exc = RuntimeError(f'Task got bad yield: {result!r}')
            self._loop.call_soon(self.__step, new_exc, context=self._context)
    finally:
        _leave_task(self._loop, self)
        self = None