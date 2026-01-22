import asyncio
import atexit
import concurrent.futures
import errno
import functools
import select
import socket
import sys
import threading
import typing
import warnings
from tornado.gen import convert_yielded
from tornado.ioloop import IOLoop, _Selectable
from typing import (
class BaseAsyncIOLoop(IOLoop):

    def initialize(self, asyncio_loop: asyncio.AbstractEventLoop, **kwargs: Any) -> None:
        self.asyncio_loop = asyncio_loop
        self.selector_loop = asyncio_loop
        if hasattr(asyncio, 'ProactorEventLoop') and isinstance(asyncio_loop, asyncio.ProactorEventLoop):
            self.selector_loop = AddThreadSelectorEventLoop(asyncio_loop)
        self.handlers: Dict[int, Tuple[Union[int, _Selectable], Callable]] = {}
        self.readers: Set[int] = set()
        self.writers: Set[int] = set()
        self.closing = False
        for loop in IOLoop._ioloop_for_asyncio.copy():
            if loop.is_closed():
                try:
                    del IOLoop._ioloop_for_asyncio[loop]
                except KeyError:
                    pass
        existing_loop = IOLoop._ioloop_for_asyncio.setdefault(asyncio_loop, self)
        if existing_loop is not self:
            raise RuntimeError(f'IOLoop {existing_loop} already associated with asyncio loop {asyncio_loop}')
        super().initialize(**kwargs)

    def close(self, all_fds: bool=False) -> None:
        self.closing = True
        for fd in list(self.handlers):
            fileobj, handler_func = self.handlers[fd]
            self.remove_handler(fd)
            if all_fds:
                self.close_fd(fileobj)
        del IOLoop._ioloop_for_asyncio[self.asyncio_loop]
        if self.selector_loop is not self.asyncio_loop:
            self.selector_loop.close()
        self.asyncio_loop.close()

    def add_handler(self, fd: Union[int, _Selectable], handler: Callable[..., None], events: int) -> None:
        fd, fileobj = self.split_fd(fd)
        if fd in self.handlers:
            raise ValueError('fd %s added twice' % fd)
        self.handlers[fd] = (fileobj, handler)
        if events & IOLoop.READ:
            self.selector_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
            self.readers.add(fd)
        if events & IOLoop.WRITE:
            self.selector_loop.add_writer(fd, self._handle_events, fd, IOLoop.WRITE)
            self.writers.add(fd)

    def update_handler(self, fd: Union[int, _Selectable], events: int) -> None:
        fd, fileobj = self.split_fd(fd)
        if events & IOLoop.READ:
            if fd not in self.readers:
                self.selector_loop.add_reader(fd, self._handle_events, fd, IOLoop.READ)
                self.readers.add(fd)
        elif fd in self.readers:
            self.selector_loop.remove_reader(fd)
            self.readers.remove(fd)
        if events & IOLoop.WRITE:
            if fd not in self.writers:
                self.selector_loop.add_writer(fd, self._handle_events, fd, IOLoop.WRITE)
                self.writers.add(fd)
        elif fd in self.writers:
            self.selector_loop.remove_writer(fd)
            self.writers.remove(fd)

    def remove_handler(self, fd: Union[int, _Selectable]) -> None:
        fd, fileobj = self.split_fd(fd)
        if fd not in self.handlers:
            return
        if fd in self.readers:
            self.selector_loop.remove_reader(fd)
            self.readers.remove(fd)
        if fd in self.writers:
            self.selector_loop.remove_writer(fd)
            self.writers.remove(fd)
        del self.handlers[fd]

    def _handle_events(self, fd: int, events: int) -> None:
        fileobj, handler_func = self.handlers[fd]
        handler_func(fileobj, events)

    def start(self) -> None:
        self.asyncio_loop.run_forever()

    def stop(self) -> None:
        self.asyncio_loop.stop()

    def call_at(self, when: float, callback: Callable, *args: Any, **kwargs: Any) -> object:
        return self.asyncio_loop.call_later(max(0, when - self.time()), self._run_callback, functools.partial(callback, *args, **kwargs))

    def remove_timeout(self, timeout: object) -> None:
        timeout.cancel()

    def add_callback(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        try:
            if asyncio.get_running_loop() is self.asyncio_loop:
                call_soon = self.asyncio_loop.call_soon
            else:
                call_soon = self.asyncio_loop.call_soon_threadsafe
        except RuntimeError:
            call_soon = self.asyncio_loop.call_soon_threadsafe
        try:
            call_soon(self._run_callback, functools.partial(callback, *args, **kwargs))
        except RuntimeError:
            pass
        except AttributeError:
            pass

    def add_callback_from_signal(self, callback: Callable, *args: Any, **kwargs: Any) -> None:
        warnings.warn('add_callback_from_signal is deprecated', DeprecationWarning)
        try:
            self.asyncio_loop.call_soon_threadsafe(self._run_callback, functools.partial(callback, *args, **kwargs))
        except RuntimeError:
            pass

    def run_in_executor(self, executor: Optional[concurrent.futures.Executor], func: Callable[..., _T], *args: Any) -> 'asyncio.Future[_T]':
        return self.asyncio_loop.run_in_executor(executor, func, *args)

    def set_default_executor(self, executor: concurrent.futures.Executor) -> None:
        return self.asyncio_loop.set_default_executor(executor)