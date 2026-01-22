import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _sock_accept(self, fut, sock):
    fd = sock.fileno()
    try:
        conn, address = sock.accept()
        conn.setblocking(False)
    except (BlockingIOError, InterruptedError):
        self._ensure_fd_no_transport(fd)
        handle = self._add_reader(fd, self._sock_accept, fut, sock)
        fut.add_done_callback(functools.partial(self._sock_read_done, fd, handle=handle))
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        fut.set_exception(exc)
    else:
        fut.set_result((conn, address))