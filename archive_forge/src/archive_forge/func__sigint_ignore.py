import contextlib
import enum
import io
import os
import signal
import subprocess
import sys
import types
import typing
from typing import Any, Optional, Type, Dict, TextIO
from autopage import command
@contextlib.contextmanager
def _sigint_ignore() -> typing.Generator[None, None, None]:
    """
    Context manager to temporarily ignore SIGINT.
    """
    old_int_handler: Any = None
    try:
        old_int_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        yield
    finally:
        if signal.getsignal(signal.SIGINT) is not None:
            signal.signal(signal.SIGINT, old_int_handler)