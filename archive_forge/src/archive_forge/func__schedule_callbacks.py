from . import events
import asyncio
import contextvars
import enum
import typing
def _schedule_callbacks(self, context: typing.Optional[contextvars.Context]=None):
    for cb in self._callbacks:
        self._loop.call_soon(cb, self, context=context if context else self._context)