from __future__ import annotations
import contextvars
import queue as stdlib_queue
import re
import sys
import threading
import time
import weakref
from functools import partial
from typing import (
import pytest
import sniffio
from .. import (
from .._core._tests.test_ki import ki_self
from .._core._tests.tutil import slow
from .._threads import (
from ..testing import wait_all_tasks_blocked
def trio_thread_fn() -> None:
    print('in Trio thread')
    assert not _core.currently_ki_protected()
    print('ki_self')
    try:
        ki_self()
    finally:
        import sys
        print('finally', sys.exc_info())