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
class SelectorThread:
    """Define ``add_reader`` methods to be called in a background select thread.

    Instances of this class start a second thread to run a selector.
    This thread is completely hidden from the user;
    all callbacks are run on the wrapped event loop's thread.

    Typically used via ``AddThreadSelectorEventLoop``,
    but can be attached to a running asyncio loop.
    """
    _closed = False

    def __init__(self, real_loop: asyncio.AbstractEventLoop) -> None:
        self._real_loop = real_loop
        self._select_cond = threading.Condition()
        self._select_args: Optional[Tuple[List[_FileDescriptorLike], List[_FileDescriptorLike]]] = None
        self._closing_selector = False
        self._thread: Optional[threading.Thread] = None
        self._thread_manager_handle = self._thread_manager()

        async def thread_manager_anext() -> None:
            await self._thread_manager_handle.__anext__()
        self._real_loop.call_soon(lambda: self._real_loop.create_task(thread_manager_anext()))
        self._readers: Dict[_FileDescriptorLike, Callable] = {}
        self._writers: Dict[_FileDescriptorLike, Callable] = {}
        self._waker_r, self._waker_w = socket.socketpair()
        self._waker_r.setblocking(False)
        self._waker_w.setblocking(False)
        _selector_loops.add(self)
        self.add_reader(self._waker_r, self._consume_waker)

    def close(self) -> None:
        if self._closed:
            return
        with self._select_cond:
            self._closing_selector = True
            self._select_cond.notify()
        self._wake_selector()
        if self._thread is not None:
            self._thread.join()
        _selector_loops.discard(self)
        self.remove_reader(self._waker_r)
        self._waker_r.close()
        self._waker_w.close()
        self._closed = True

    async def _thread_manager(self) -> typing.AsyncGenerator[None, None]:
        self._thread = threading.Thread(name='Tornado selector', daemon=True, target=self._run_select)
        self._thread.start()
        self._start_select()
        try:
            yield
        except GeneratorExit:
            self.close()
            raise

    def _wake_selector(self) -> None:
        if self._closed:
            return
        try:
            self._waker_w.send(b'a')
        except BlockingIOError:
            pass

    def _consume_waker(self) -> None:
        try:
            self._waker_r.recv(1024)
        except BlockingIOError:
            pass

    def _start_select(self) -> None:
        with self._select_cond:
            assert self._select_args is None
            self._select_args = (list(self._readers.keys()), list(self._writers.keys()))
            self._select_cond.notify()

    def _run_select(self) -> None:
        while True:
            with self._select_cond:
                while self._select_args is None and (not self._closing_selector):
                    self._select_cond.wait()
                if self._closing_selector:
                    return
                assert self._select_args is not None
                to_read, to_write = self._select_args
                self._select_args = None
            try:
                rs, ws, xs = select.select(to_read, to_write, to_write)
                ws = ws + xs
            except OSError as e:
                if e.errno == getattr(errno, 'WSAENOTSOCK', errno.EBADF):
                    rs, _, _ = select.select([self._waker_r.fileno()], [], [], 0)
                    if rs:
                        ws = []
                    else:
                        raise
                else:
                    raise
            try:
                self._real_loop.call_soon_threadsafe(self._handle_select, rs, ws)
            except RuntimeError:
                pass
            except AttributeError:
                pass

    def _handle_select(self, rs: List[_FileDescriptorLike], ws: List[_FileDescriptorLike]) -> None:
        for r in rs:
            self._handle_event(r, self._readers)
        for w in ws:
            self._handle_event(w, self._writers)
        self._start_select()

    def _handle_event(self, fd: _FileDescriptorLike, cb_map: Dict[_FileDescriptorLike, Callable]) -> None:
        try:
            callback = cb_map[fd]
        except KeyError:
            return
        callback()

    def add_reader(self, fd: _FileDescriptorLike, callback: Callable[..., None], *args: Any) -> None:
        self._readers[fd] = functools.partial(callback, *args)
        self._wake_selector()

    def add_writer(self, fd: _FileDescriptorLike, callback: Callable[..., None], *args: Any) -> None:
        self._writers[fd] = functools.partial(callback, *args)
        self._wake_selector()

    def remove_reader(self, fd: _FileDescriptorLike) -> bool:
        try:
            del self._readers[fd]
        except KeyError:
            return False
        self._wake_selector()
        return True

    def remove_writer(self, fd: _FileDescriptorLike) -> bool:
        try:
            del self._writers[fd]
        except KeyError:
            return False
        self._wake_selector()
        return True