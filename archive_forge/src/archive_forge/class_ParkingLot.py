from __future__ import annotations
import math
from collections import OrderedDict
from typing import TYPE_CHECKING
import attrs
from .. import _core
from .._util import final
@final
@attrs.define(eq=False, hash=False)
class ParkingLot:
    """A fair wait queue with cancellation and requeueing.

    This class encapsulates the tricky parts of implementing a wait
    queue. It's useful for implementing higher-level synchronization
    primitives like queues and locks.

    In addition to the methods below, you can use ``len(parking_lot)`` to get
    the number of parked tasks, and ``if parking_lot: ...`` to check whether
    there are any parked tasks.

    """
    _parked: OrderedDict[Task, None] = attrs.field(factory=OrderedDict, init=False)

    def __len__(self) -> int:
        """Returns the number of parked tasks."""
        return len(self._parked)

    def __bool__(self) -> bool:
        """True if there are parked tasks, False otherwise."""
        return bool(self._parked)

    @_core.enable_ki_protection
    async def park(self) -> None:
        """Park the current task until woken by a call to :meth:`unpark` or
        :meth:`unpark_all`.

        """
        task = _core.current_task()
        self._parked[task] = None
        task.custom_sleep_data = self

        def abort_fn(_: _core.RaiseCancelT) -> _core.Abort:
            del task.custom_sleep_data._parked[task]
            return _core.Abort.SUCCEEDED
        await _core.wait_task_rescheduled(abort_fn)

    def _pop_several(self, count: int | float) -> Iterator[Task]:
        if isinstance(count, float):
            if math.isinf(count):
                count = len(self._parked)
            else:
                raise ValueError('Cannot pop a non-integer number of tasks.')
        else:
            count = min(count, len(self._parked))
        for _ in range(count):
            task, _ = self._parked.popitem(last=False)
            yield task

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

    def unpark_all(self) -> list[Task]:
        """Unpark all parked tasks."""
        return self.unpark(count=len(self))

    @_core.enable_ki_protection
    def repark(self, new_lot: ParkingLot, *, count: int | float=1) -> None:
        """Move parked tasks from one :class:`ParkingLot` object to another.

        This dequeues ``count`` tasks from one lot, and requeues them on
        another, preserving order. For example::

           async def parker(lot):
               print("sleeping")
               await lot.park()
               print("woken")

           async def main():
               lot1 = trio.lowlevel.ParkingLot()
               lot2 = trio.lowlevel.ParkingLot()
               async with trio.open_nursery() as nursery:
                   nursery.start_soon(parker, lot1)
                   await trio.testing.wait_all_tasks_blocked()
                   assert len(lot1) == 1
                   assert len(lot2) == 0
                   lot1.repark(lot2)
                   assert len(lot1) == 0
                   assert len(lot2) == 1
                   # This wakes up the task that was originally parked in lot1
                   lot2.unpark()

        If there are fewer than ``count`` tasks parked, then reparks as many
        tasks as are available and then returns successfully.

        Args:
          new_lot (ParkingLot): the parking lot to move tasks to.
          count (int|math.inf): the number of tasks to move.

        """
        if not isinstance(new_lot, ParkingLot):
            raise TypeError('new_lot must be a ParkingLot')
        for task in self._pop_several(count):
            new_lot._parked[task] = None
            task.custom_sleep_data = new_lot

    def repark_all(self, new_lot: ParkingLot) -> None:
        """Move all parked tasks from one :class:`ParkingLot` object to
        another.

        See :meth:`repark` for details.

        """
        return self.repark(new_lot, count=len(self))

    def statistics(self) -> ParkingLotStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``tasks_waiting``: The number of tasks blocked on this lot's
          :meth:`park` method.

        """
        return ParkingLotStatistics(tasks_waiting=len(self._parked))