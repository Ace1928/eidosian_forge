from __future__ import annotations
import contextlib
import select
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Literal
import attrs
from .. import _core
from ._io_common import wake_all
from ._run import Task, _public
from ._wakeup_socketpair import WakeupSocketpair
@attrs.define(eq=False)
class EpollWaiters:
    read_task: Task | None = None
    write_task: Task | None = None
    current_flags: int = 0