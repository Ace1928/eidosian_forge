import importlib
import inspect
import os
import warnings
from eventlet import patcher
from eventlet.support import greenlets as greenlet
from eventlet import timeout
def trampoline(fd, read=None, write=None, timeout=None, timeout_exc=timeout.Timeout, mark_as_closed=None):
    """Suspend the current coroutine until the given socket object or file
    descriptor is ready to *read*, ready to *write*, or the specified
    *timeout* elapses, depending on arguments specified.

    To wait for *fd* to be ready to read, pass *read* ``=True``; ready to
    write, pass *write* ``=True``. To specify a timeout, pass the *timeout*
    argument in seconds.

    If the specified *timeout* elapses before the socket is ready to read or
    write, *timeout_exc* will be raised instead of ``trampoline()``
    returning normally.

    .. note :: |internal|
    """
    t = None
    hub = get_hub()
    current = greenlet.getcurrent()
    if hub.greenlet is current:
        raise RuntimeError('do not call blocking functions from the mainloop')
    if read and write:
        raise RuntimeError('not allowed to trampoline for reading and writing')
    try:
        fileno = fd.fileno()
    except AttributeError:
        fileno = fd
    if timeout is not None:

        def _timeout(exc):
            current.throw(exc)
        t = hub.schedule_call_global(timeout, _timeout, timeout_exc)
    try:
        if read:
            listener = hub.add(hub.READ, fileno, current.switch, current.throw, mark_as_closed)
        elif write:
            listener = hub.add(hub.WRITE, fileno, current.switch, current.throw, mark_as_closed)
        try:
            return hub.switch()
        finally:
            hub.remove(listener)
    finally:
        if t is not None:
            t.cancel()