from __future__ import annotations
import math
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING
import trio
class TooSlowError(Exception):
    """Raised by :func:`fail_after` and :func:`fail_at` if the timeout
    expires.

    """