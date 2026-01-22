from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
def _schedule_pending_tasks(self) -> None:
    """Schedules all pending asynchronous tasks that were created before
        the nursery to be executed on the nursery soon.
        """
    for task, scope, args in self._pending_tasks:
        self._nursery.start_soon(task, scope, *args)
    del self._pending_tasks[:]