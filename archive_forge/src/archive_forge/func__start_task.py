from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
def _start_task(self, task: Callable[Concatenate[trio.CancelScope, _Spec], Awaitable], *args: _Spec.args) -> trio.CancelScope:
    """Starts an asynchronous task in the Trio nursery managed by the
        main loop. If the nursery has not started yet, store a reference to
        the task and the arguments so we can start the task when the nursery
        is open.

        Parameters:
            task: a Trio task to run

        Returns:
            a cancellation scope for the Trio task
        """
    scope = trio.CancelScope()
    if self._nursery:
        self._nursery.start_soon(task, scope, *args)
    else:
        self._pending_tasks.append((task, scope, args))
    return scope