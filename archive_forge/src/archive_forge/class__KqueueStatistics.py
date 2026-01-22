from __future__ import annotations
import errno
import select
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, Literal
import attrs
import outcome
from .. import _core
from ._run import _public
from ._wakeup_socketpair import WakeupSocketpair
@attrs.frozen(eq=False)
class _KqueueStatistics:
    tasks_waiting: int
    monitors: int
    backend: Literal['kqueue'] = attrs.field(init=False, default='kqueue')