import asyncio
import concurrent.futures
import errno
import os
import sys
import socket
import ssl
import stat
from tornado.concurrent import dummy_executor, run_on_executor
from tornado.ioloop import IOLoop
from tornado.util import Configurable, errno_from_exception
from typing import List, Callable, Any, Type, Dict, Union, Tuple, Awaitable, Optional
def add_accept_handler(sock: socket.socket, callback: Callable[[socket.socket, Any], None]) -> Callable[[], None]:
    """Adds an `.IOLoop` event handler to accept new connections on ``sock``.

    When a connection is accepted, ``callback(connection, address)`` will
    be run (``connection`` is a socket object, and ``address`` is the
    address of the other end of the connection).  Note that this signature
    is different from the ``callback(fd, events)`` signature used for
    `.IOLoop` handlers.

    A callable is returned which, when called, will remove the `.IOLoop`
    event handler and stop processing further incoming connections.

    .. versionchanged:: 5.0
       The ``io_loop`` argument (deprecated since version 4.1) has been removed.

    .. versionchanged:: 5.0
       A callable is returned (``None`` was returned before).
    """
    io_loop = IOLoop.current()
    removed = [False]

    def accept_handler(fd: socket.socket, events: int) -> None:
        for i in range(_DEFAULT_BACKLOG):
            if removed[0]:
                return
            try:
                connection, address = sock.accept()
            except BlockingIOError:
                return
            except ConnectionAbortedError:
                continue
            callback(connection, address)

    def remove_handler() -> None:
        io_loop.remove_handler(sock)
        removed[0] = True
    io_loop.add_handler(sock, accept_handler, IOLoop.READ)
    return remove_handler