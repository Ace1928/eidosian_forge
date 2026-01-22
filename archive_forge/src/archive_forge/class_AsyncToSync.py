import asyncio
import asyncio.coroutines
import contextvars
import functools
import inspect
import os
import sys
import threading
import warnings
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from typing import (
from .current_thread_executor import CurrentThreadExecutor
from .local import Local
class AsyncToSync(Generic[_P, _R]):
    """
    Utility class which turns an awaitable that only works on the thread with
    the event loop into a synchronous callable that works in a subthread.

    If the call stack contains an async loop, the code runs there.
    Otherwise, the code runs in a new loop in a new thread.

    Either way, this thread then pauses and waits to run any thread_sensitive
    code called from further down the call stack using SyncToAsync, before
    finally exiting once the async task returns.
    """
    executors: 'Local' = Local()
    loop_thread_executors: 'Dict[asyncio.AbstractEventLoop, CurrentThreadExecutor]' = {}

    def __init__(self, awaitable: Union[Callable[_P, Coroutine[Any, Any, _R]], Callable[_P, Awaitable[_R]]], force_new_loop: bool=False):
        if not callable(awaitable) or (not iscoroutinefunction(awaitable) and (not iscoroutinefunction(getattr(awaitable, '__call__', awaitable)))):
            warnings.warn('async_to_sync was passed a non-async-marked callable', stacklevel=2)
        self.awaitable = awaitable
        try:
            self.__self__ = self.awaitable.__self__
        except AttributeError:
            pass
        self.force_new_loop = force_new_loop
        self.main_event_loop = None
        try:
            self.main_event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        __traceback_hide__ = True
        if not self.force_new_loop and (not self.main_event_loop):
            main_event_loop_pid = getattr(SyncToAsync.threadlocal, 'main_event_loop_pid', None)
            if main_event_loop_pid and main_event_loop_pid == os.getpid():
                self.main_event_loop = getattr(SyncToAsync.threadlocal, 'main_event_loop', None)
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if event_loop.is_running():
                raise RuntimeError('You cannot use AsyncToSync in the same thread as an async event loop - just await the async function directly.')
        call_result: 'Future[_R]' = Future()
        old_executor = getattr(self.executors, 'current', None)
        current_executor = CurrentThreadExecutor()
        self.executors.current = current_executor
        context = [contextvars.copy_context()]
        task_context = getattr(SyncToAsync.threadlocal, 'task_context', None)
        loop = None
        try:
            awaitable = self.main_wrap(call_result, sys.exc_info(), task_context, context, *args, **kwargs)
            if not (self.main_event_loop and self.main_event_loop.is_running()):
                loop = asyncio.new_event_loop()
                self.loop_thread_executors[loop] = current_executor
                loop_executor = ThreadPoolExecutor(max_workers=1)
                loop_future = loop_executor.submit(self._run_event_loop, loop, awaitable)
                if current_executor:
                    current_executor.run_until_future(loop_future)
                loop_future.result()
            else:
                self.main_event_loop.call_soon_threadsafe(self.main_event_loop.create_task, awaitable)
                if current_executor:
                    current_executor.run_until_future(call_result)
        finally:
            if loop is not None:
                del self.loop_thread_executors[loop]
            _restore_context(context[0])
            self.executors.current = old_executor
        return call_result.result()

    def _run_event_loop(self, loop, coro):
        """
        Runs the given event loop (designed to be called in a thread).
        """
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(coro)
        finally:
            try:
                tasks = asyncio.all_tasks(loop)
                for task in tasks:
                    task.cancel()

                async def gather():
                    await asyncio.gather(*tasks, return_exceptions=True)
                loop.run_until_complete(gather())
                for task in tasks:
                    if task.cancelled():
                        continue
                    if task.exception() is not None:
                        loop.call_exception_handler({'message': 'unhandled exception during loop shutdown', 'exception': task.exception(), 'task': task})
                if hasattr(loop, 'shutdown_asyncgens'):
                    loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                asyncio.set_event_loop(self.main_event_loop)

    def __get__(self, parent: Any, objtype: Any) -> Callable[_P, _R]:
        """
        Include self for methods
        """
        func = functools.partial(self.__call__, parent)
        return functools.update_wrapper(func, self.awaitable)

    async def main_wrap(self, call_result: 'Future[_R]', exc_info: 'OptExcInfo', task_context: 'Optional[List[asyncio.Task[Any]]]', context: List[contextvars.Context], *args: _P.args, **kwargs: _P.kwargs) -> None:
        """
        Wraps the awaitable with something that puts the result into the
        result/exception future.
        """
        __traceback_hide__ = True
        if context is not None:
            _restore_context(context[0])
        current_task = asyncio.current_task()
        if current_task is not None and task_context is not None:
            task_context.append(current_task)
        try:
            if exc_info[1]:
                try:
                    raise exc_info[1]
                except BaseException:
                    result = await self.awaitable(*args, **kwargs)
            else:
                result = await self.awaitable(*args, **kwargs)
        except BaseException as e:
            call_result.set_exception(e)
        else:
            call_result.set_result(result)
        finally:
            if current_task is not None and task_context is not None:
                task_context.remove(current_task)
            context[0] = contextvars.copy_context()