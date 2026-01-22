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
def after_io_wait(self, timeout: float) -> None:
    if self.primed:
        print('instrument triggered')
        in_host(lambda: cscope.cancel())
        trio.lowlevel.remove_instrument(self)