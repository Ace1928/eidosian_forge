from __future__ import annotations
from typing import TYPE_CHECKING, Generic, TypeVar
import attrs
from .. import _core
from .._deprecate import deprecated
from .._util import final
@attrs.frozen
class UnboundedQueueStatistics:
    """An object containing debugging information.

    Currently, the following fields are defined:

    * ``qsize``: The number of items currently in the queue.
    * ``tasks_waiting``: The number of tasks blocked on this queue's
      :meth:`get_batch` method.

    """
    qsize: int
    tasks_waiting: int