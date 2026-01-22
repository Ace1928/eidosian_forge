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
def _sock_sendall(self, fut, sock, view, pos):
    if fut.done():
        return
    start = pos[0]
    try:
        n = sock.send(view[start:])
    except (BlockingIOError, InterruptedError):
        return
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        fut.set_exception(exc)
        return
    start += n
    if start == len(view):
        fut.set_result(None)
    else:
        pos[0] = start