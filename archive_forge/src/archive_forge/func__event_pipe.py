import asyncio
import atexit
import contextvars
import io
import os
import sys
import threading
import traceback
import warnings
from binascii import b2a_hex
from collections import defaultdict, deque
from io import StringIO, TextIOBase
from threading import local
from typing import Any, Callable, Deque, Dict, Optional
import zmq
from jupyter_client.session import extract_header
from tornado.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
@property
def _event_pipe(self):
    """thread-local event pipe for signaling events that should be processed in the thread"""
    try:
        event_pipe = self._local.event_pipe
    except AttributeError:
        ctx = self.socket.context
        event_pipe = ctx.socket(zmq.PUSH)
        event_pipe.linger = 0
        event_pipe.connect(self._event_interface)
        self._local.event_pipe = event_pipe
        with self._event_pipe_gc_lock:
            self._event_pipes[threading.current_thread()] = event_pipe
    return event_pipe