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
def test_run_fn_as_system_task_catched_badly_typed_token() -> None:
    with pytest.raises(RuntimeError):
        from_thread_run_sync(_core.current_time, trio_token='Not TrioTokentype')