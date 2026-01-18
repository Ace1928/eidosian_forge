from asyncio import Future, Queue, ensure_future, sleep
from inspect import isawaitable
from typing import Any, AsyncIterator, Callable, Optional, Set
Emit an event.