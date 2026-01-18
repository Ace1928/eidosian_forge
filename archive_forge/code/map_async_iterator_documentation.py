from asyncio import CancelledError, Event, Task, ensure_future, wait
from concurrent.futures import FIRST_COMPLETED
from inspect import isasyncgen, isawaitable
from typing import cast, Any, AsyncIterable, Callable, Optional, Set, Type, Union
from types import TracebackType
Mark the iterator as closed.