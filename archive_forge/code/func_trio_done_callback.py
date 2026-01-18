from __future__ import annotations
import asyncio
import contextlib
import contextvars
import queue
import signal
import socket
import sys
import threading
import time
import traceback
import warnings
from functools import partial
from math import inf
from typing import (
import pytest
from outcome import Outcome
import trio
import trio.testing
from trio.abc import Instrument
from ..._util import signal_raise
from .tutil import gc_collect_harder, restore_unraisablehook
def trio_done_callback(main_outcome: Outcome[object]) -> None:
    print(f'trio_fn finished: {main_outcome!r}')
    trio_done_fut.set_result(main_outcome)