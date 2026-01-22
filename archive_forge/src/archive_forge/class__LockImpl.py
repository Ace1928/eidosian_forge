from __future__ import annotations
import math
from typing import TYPE_CHECKING, Protocol
import attrs
import trio
from . import _core
from ._core import Abort, ParkingLot, RaiseCancelT, enable_ki_protection
from ._util import final
@attrs.define(eq=False, hash=False, repr=False, slots=False)
class _LockImpl(AsyncContextManagerMixin):
    _lot: ParkingLot = attrs.field(factory=ParkingLot, init=False)
    _owner: Task | None = attrs.field(default=None, init=False)

    def __repr__(self) -> str:
        if self.locked():
            s1 = 'locked'
            s2 = f' with {len(self._lot)} waiters'
        else:
            s1 = 'unlocked'
            s2 = ''
        return f'<{s1} {self.__class__.__name__} object at {id(self):#x}{s2}>'

    def locked(self) -> bool:
        """Check whether the lock is currently held.

        Returns:
          bool: True if the lock is held, False otherwise.

        """
        return self._owner is not None

    @enable_ki_protection
    def acquire_nowait(self) -> None:
        """Attempt to acquire the lock, without blocking.

        Raises:
          WouldBlock: if the lock is held.

        """
        task = trio.lowlevel.current_task()
        if self._owner is task:
            raise RuntimeError('attempt to re-acquire an already held Lock')
        elif self._owner is None and (not self._lot):
            self._owner = task
        else:
            raise trio.WouldBlock

    @enable_ki_protection
    async def acquire(self) -> None:
        """Acquire the lock, blocking if necessary."""
        await trio.lowlevel.checkpoint_if_cancelled()
        try:
            self.acquire_nowait()
        except trio.WouldBlock:
            await self._lot.park()
        else:
            await trio.lowlevel.cancel_shielded_checkpoint()

    @enable_ki_protection
    def release(self) -> None:
        """Release the lock.

        Raises:
          RuntimeError: if the calling task does not hold the lock.

        """
        task = trio.lowlevel.current_task()
        if task is not self._owner:
            raise RuntimeError("can't release a Lock you don't own")
        if self._lot:
            self._owner, = self._lot.unpark(count=1)
        else:
            self._owner = None

    def statistics(self) -> LockStatistics:
        """Return an object containing debugging information.

        Currently the following fields are defined:

        * ``locked``: boolean indicating whether the lock is held.
        * ``owner``: the :class:`trio.lowlevel.Task` currently holding the lock,
          or None if the lock is not held.
        * ``tasks_waiting``: The number of tasks blocked on this lock's
          :meth:`acquire` method.

        """
        return LockStatistics(locked=self.locked(), owner=self._owner, tasks_waiting=len(self._lot))