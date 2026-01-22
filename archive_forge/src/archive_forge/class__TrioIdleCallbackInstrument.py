from __future__ import annotations
import logging
import typing
import exceptiongroup
import trio
from .abstract_loop import EventLoop, ExitMainLoop
class _TrioIdleCallbackInstrument(trio.abc.Instrument):
    """IDLE callbacks emulation helper."""
    __slots__ = ('idle_callbacks',)

    def __init__(self, idle_callbacks: Mapping[Hashable, Callable[[], typing.Any]]):
        self.idle_callbacks = idle_callbacks

    def before_io_wait(self, timeout: float) -> None:
        if timeout > 0:
            for idle_callback in self.idle_callbacks.values():
                idle_callback()