from __future__ import annotations
import os
import pickle
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from typing import TypeVar, cast
from ._core._eventloop import current_time, get_async_backend, get_cancelled_exc_class
from ._core._exceptions import BrokenWorkerProcess
from ._core._subprocesses import open_process
from ._core._synchronization import CapacityLimiter
from ._core._tasks import CancelScope, fail_after
from .abc import ByteReceiveStream, ByteSendStream, Process
from .lowlevel import RunVar, checkpoint_if_cancelled
from .streams.buffered import BufferedByteReceiveStream

    Return the capacity limiter that is used by default to limit the number of worker
    processes.

    :return: a capacity limiter object

    