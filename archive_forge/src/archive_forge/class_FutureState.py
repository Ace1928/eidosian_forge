from . import events
import asyncio
import contextvars
import enum
import typing
class FutureState(enum.Enum):
    PENDING = enum.auto()
    CANCELLED = enum.auto()
    DONE_WITH_RESULT = enum.auto()
    DONE_WITH_EXCEPTION = enum.auto()