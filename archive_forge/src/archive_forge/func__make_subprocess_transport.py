import errno
import os
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import futures
from . import selector_events
from . import selectors
from . import transports
from .coroutines import coroutine
from .log import logger
@coroutine
def _make_subprocess_transport(self, protocol, args, shell, stdin, stdout, stderr, bufsize, extra=None, **kwargs):
    with events.get_child_watcher() as watcher:
        waiter = futures.Future(loop=self)
        transp = _UnixSubprocessTransport(self, protocol, args, shell, stdin, stdout, stderr, bufsize, waiter=waiter, extra=extra, **kwargs)
        watcher.add_child_handler(transp.get_pid(), self._child_watcher_callback, transp)
        try:
            yield from waiter
        except Exception as exc:
            err = exc
        else:
            err = None
        if err is not None:
            transp.close()
            yield from transp._wait()
            raise err
    return transp