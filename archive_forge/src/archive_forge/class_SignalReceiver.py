from __future__ import annotations
import signal
from collections import OrderedDict
from contextlib import contextmanager
from typing import TYPE_CHECKING
import trio
from ._util import ConflictDetector, is_main_thread, signal_raise
class SignalReceiver:

    def __init__(self) -> None:
        self._pending: OrderedDict[int, None] = OrderedDict()
        self._lot = trio.lowlevel.ParkingLot()
        self._conflict_detector = ConflictDetector('only one task can iterate on a signal receiver at a time')
        self._closed = False

    def _add(self, signum: int) -> None:
        if self._closed:
            signal_raise(signum)
        else:
            self._pending[signum] = None
            self._lot.unpark()

    def _redeliver_remaining(self) -> None:
        self._closed = True

        def deliver_next() -> None:
            if self._pending:
                signum, _ = self._pending.popitem(last=False)
                try:
                    signal_raise(signum)
                finally:
                    deliver_next()
        deliver_next()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> int:
        if self._closed:
            raise RuntimeError('open_signal_receiver block already exited')
        with self._conflict_detector:
            if not self._pending:
                await self._lot.park()
            else:
                await trio.lowlevel.checkpoint()
            signum, _ = self._pending.popitem(last=False)
            return signum