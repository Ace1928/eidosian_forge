from __future__ import unicode_literals
from ..terminal.vt100_input import InputStream
from .asyncio_base import AsyncioTimeout
from .base import EventLoop, INPUT_TIMEOUT
from .callbacks import EventLoopCallbacks
from .posix_utils import PosixStdinReader
import asyncio
import signal
def timeout_handler():
    """
                When no input has been received for INPUT_TIMEOUT seconds,
                flush the input stream and fire the timeout event.
                """
    inputstream.flush()
    callbacks.input_timeout()