from __future__ import annotations
from threading import Thread, current_thread
from typing import Any, Callable, List, Optional, TypeVar
from typing_extensions import ParamSpec, Protocol, TypedDict
from twisted._threads import pool as _pool
from twisted.python import context, log
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.python.versions import Version
@property
def working(self) -> list[None]:
    """
        For legacy compatibility purposes, return the number of busy workers as
        expressed by a list the length of that number.

        @return: the number of workers currently processing a work item.
        @rtype: L{list} of L{None}
        """
    return [None] * self._team.statistics().busyWorkerCount