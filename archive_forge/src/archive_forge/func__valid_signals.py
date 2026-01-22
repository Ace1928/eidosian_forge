import logging
import os
import re
import signal
import sys
import threading
from subprocess import call
from types import FrameType
from typing import Any, Callable, Dict, List, Set, Union
import pytorch_lightning as pl
from lightning_fabric.plugins.environments import SLURMEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS, _PYTHON_GREATER_EQUAL_3_8_0
from pytorch_lightning.utilities.rank_zero import rank_prefixed_message, rank_zero_info
@staticmethod
def _valid_signals() -> Set[signal.Signals]:
    """Returns all valid signals supported on the current platform.

        Behaves identically to :func:`signals.valid_signals` in Python 3.8+ and implements the equivalent behavior for
        older Python versions.

        """
    if _PYTHON_GREATER_EQUAL_3_8_0:
        return signal.valid_signals()
    if _IS_WINDOWS:
        return {signal.SIGABRT, signal.SIGFPE, signal.SIGILL, signal.SIGINT, signal.SIGSEGV, signal.SIGTERM, signal.SIGBREAK}
    return set(signal.Signals)