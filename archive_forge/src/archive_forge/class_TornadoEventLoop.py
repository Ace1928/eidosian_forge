from __future__ import annotations
import asyncio
import functools
import logging
import typing
from contextlib import suppress
from tornado import ioloop
from .abstract_loop import EventLoop, ExitMainLoop
class TornadoEventLoop(EventLoop):
    """This is an Urwid-specific event loop to plug into its MainLoop.
    It acts as an adaptor for Tornado's IOLoop which does all
    heavy lifting except idle-callbacks.
    """

    def __init__(self, loop: ioloop.IOLoop | None=None) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__).getChild(self.__class__.__name__)
        if loop:
            self._loop: ioloop.IOLoop = loop
        else:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            self._loop = ioloop.IOLoop.current()
        self._pending_alarms: dict[object, int] = {}
        self._watch_handles: dict[int, int] = {}
        self._max_watch_handle: int = 0
        self._exc: BaseException | None = None
        self._idle_asyncio_handle: object | None = None
        self._idle_handle: int = 0
        self._idle_callbacks: dict[int, Callable[[], typing.Any]] = {}

    def _also_call_idle(self, callback: Callable[_Spec, _T]) -> Callable[_Spec, _T]:
        """
        Wrap the callback to also call _entering_idle.
        """

        @functools.wraps(callback)
        def wrapper(*args: _Spec.args, **kwargs: _Spec.kwargs) -> _T:
            if not self._idle_asyncio_handle:
                self._idle_asyncio_handle = self._loop.call_later(0, self._entering_idle)
            return callback(*args, **kwargs)
        return wrapper

    def _entering_idle(self) -> None:
        """
        Call all the registered idle callbacks.
        """
        try:
            for callback in self._idle_callbacks.values():
                callback()
        finally:
            self._idle_asyncio_handle = None

    def run_in_executor(self, executor: Executor, func: Callable[_Spec, _T], *args: _Spec.args, **kwargs: _Spec.kwargs) -> asyncio.Future[_T]:
        """Run callable in executor.

        :param executor: Executor to use for running the function
        :type executor: concurrent.futures.Executor
        :param func: function to call
        :type func: Callable
        :param args: arguments to function (positional only)
        :type args: object
        :param kwargs: keyword arguments to function (keyword only)
        :type kwargs: object
        :return: future object for the function call outcome.
        :rtype: asyncio.Future
        """
        return self._loop.run_in_executor(executor, functools.partial(func, *args, **kwargs))

    def alarm(self, seconds: float, callback: Callable[[], typing.Any]):

        @self._also_call_idle
        @functools.wraps(callback)
        def wrapped() -> None:
            with suppress(KeyError):
                del self._pending_alarms[handle]
            self.handle_exit(callback)()
        handle = self._loop.add_timeout(self._loop.time() + seconds, wrapped)
        self._pending_alarms[handle] = 1
        return handle

    def remove_alarm(self, handle: object) -> bool:
        self._loop.remove_timeout(handle)
        try:
            del self._pending_alarms[handle]
        except KeyError:
            return False
        return True

    def watch_file(self, fd: int, callback: Callable[[], _T]) -> int:

        @self._also_call_idle
        def handler(_fd: int, _events: int) -> None:
            self.handle_exit(callback)()
        self._loop.add_handler(fd, handler, ioloop.IOLoop.READ)
        self._max_watch_handle += 1
        handle = self._max_watch_handle
        self._watch_handles[handle] = fd
        return handle

    def remove_watch_file(self, handle: int) -> bool:
        fd = self._watch_handles.pop(handle, None)
        if fd is None:
            return False
        self._loop.remove_handler(fd)
        return True

    def enter_idle(self, callback: Callable[[], typing.Any]) -> int:
        """
        Add a callback for entering idle.

        Returns a handle that may be passed to remove_idle()
        """
        self._idle_handle += 1
        self._idle_callbacks[self._idle_handle] = callback
        return self._idle_handle

    def remove_enter_idle(self, handle: int) -> bool:
        """
        Remove an idle callback.

        Returns True if the handle was removed.
        """
        try:
            del self._idle_callbacks[handle]
        except KeyError:
            return False
        return True

    def handle_exit(self, f: Callable[_Spec, _T]) -> Callable[_Spec, _T | Literal[False]]:

        @functools.wraps(f)
        def wrapper(*args: _Spec.args, **kwargs: _Spec.kwargs) -> _T | Literal[False]:
            try:
                return f(*args, **kwargs)
            except ExitMainLoop:
                pass
            except Exception as exc:
                self._exc = exc
            if self._idle_asyncio_handle:
                self._loop.remove_timeout(self._idle_asyncio_handle)
                self._idle_asyncio_handle = None
            self._loop.stop()
            return False
        return wrapper

    def run(self) -> None:
        self._loop.start()
        if self._exc:
            exc, self._exc = (self._exc, None)
            raise exc.with_traceback(exc.__traceback__)