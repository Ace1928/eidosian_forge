from __future__ import annotations
import dataclasses
import datetime
import functools
import os
import signal
import time
import typing as t
from .io import (
from .config import (
from .util import (
from .thread import (
from .constants import (
from .test import (
def timeout_waiter(timeout_seconds: int) -> None:
    """Background thread which will kill the current process if the timeout elapses."""
    time.sleep(timeout_seconds)
    os.kill(os.getpid(), signal.SIGUSR1)