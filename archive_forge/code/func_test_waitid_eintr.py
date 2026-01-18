from __future__ import annotations
import gc
import os
import random
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path as SyncPath
from signal import Signals
from typing import (
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import (
from .._core._tests.tutil import skip_if_fbsd_pipes_broken, slow
from ..lowlevel import open_process
from ..testing import MockClock, assert_no_checkpoints, wait_all_tasks_blocked
@slow
def test_waitid_eintr() -> None:
    from .._subprocess_platform import wait_child_exiting
    if TYPE_CHECKING and (sys.platform == 'win32' or sys.platform == 'darwin'):
        return
    if not wait_child_exiting.__module__.endswith('waitid'):
        pytest.skip('waitid only')
    from .._subprocess_platform.waitid import sync_wait_reapable
    got_alarm = False
    sleeper = subprocess.Popen(['sleep', '3600'])

    def on_alarm(sig: int, frame: FrameType | None) -> None:
        nonlocal got_alarm
        got_alarm = True
        sleeper.kill()
    old_sigalrm = signal.signal(signal.SIGALRM, on_alarm)
    try:
        signal.alarm(1)
        sync_wait_reapable(sleeper.pid)
        assert sleeper.wait(timeout=1) == -9
    finally:
        if sleeper.returncode is None:
            sleeper.kill()
            sleeper.wait()
        signal.signal(signal.SIGALRM, old_sigalrm)