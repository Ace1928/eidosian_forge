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
def SLEEP(seconds: int) -> list[str]:
    return python(f'import time; time.sleep({seconds})')