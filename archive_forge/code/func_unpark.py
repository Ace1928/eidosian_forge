from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attrs
from .. import _core
from .._util import final
@_core.enable_ki_protection
def unpark(self, *, count: int | float=1) -> list[Task]:
    """Unpark one or more tasks.

        This wakes up ``count`` tasks that are blocked in :meth:`park`. If
        there are fewer than ``count`` tasks parked, then wakes as many tasks
        are available and then returns successfully.

        Args:
          count (int | math.inf): the number of tasks to unpark.

        """
    tasks = list(self._pop_several(count))
    for task in tasks:
        _core.reschedule(task)
    return tasks