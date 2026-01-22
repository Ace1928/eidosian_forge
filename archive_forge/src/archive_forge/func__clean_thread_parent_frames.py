import asyncio
import builtins
import gc
import getpass
import os
import signal
import sys
import threading
import typing as t
from contextlib import contextmanager
from functools import partial
import comm
from IPython.core import release
from IPython.utils.tokenutil import line_at_cursor, token_at_cursor
from jupyter_client.session import extract_header
from traitlets import Any, Bool, HasTraits, Instance, List, Type, observe, observe_compat
from zmq.eventloop.zmqstream import ZMQStream
from .comm.comm import BaseComm
from .comm.manager import CommManager
from .compiler import XCachingCompiler
from .eventloops import _use_appnope
from .iostream import OutStream
from .kernelbase import Kernel as KernelBase
from .kernelbase import _accepts_parameters
from .zmqshell import ZMQInteractiveShell
def _clean_thread_parent_frames(self, phase: t.Literal['start', 'stop'], info: t.Dict[str, t.Any]):
    """Clean parent frames of threads which are no longer running.
        This is meant to be invoked by garbage collector callback hook.

        The implementation enumerates the threads because there is no "exit" hook yet,
        but there might be one in the future: https://bugs.python.org/issue14073

        This is a no-op if the `self._stdout` and `self._stderr` are not
        sub-classes of `OutStream`.
        """
    if phase != 'start':
        return
    active_threads = {thread.ident for thread in threading.enumerate()}
    for stream in [self._stdout, self._stderr]:
        if isinstance(stream, OutStream):
            thread_to_parent_header = stream._thread_to_parent_header
            for identity in list(thread_to_parent_header.keys()):
                if identity not in active_threads:
                    try:
                        del thread_to_parent_header[identity]
                    except KeyError:
                        pass
            thread_to_parent = stream._thread_to_parent
            for identity in list(thread_to_parent.keys()):
                if identity not in active_threads:
                    try:
                        del thread_to_parent[identity]
                    except KeyError:
                        pass