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
class BadCapacityLimiter:

    async def acquire_on_behalf_of(self, borrower: Task) -> None:
        record.append('acquire')

    def release_on_behalf_of(self, borrower: Task) -> NoReturn:
        record.append('release')
        raise ValueError('release on behalf')