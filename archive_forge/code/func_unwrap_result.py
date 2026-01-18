from __future__ import annotations
import warnings
from asyncio import Future
from collections import deque
from functools import partial
from itertools import chain
from typing import Any, Awaitable, Callable, NamedTuple, TypeVar, cast, overload
import zmq as _zmq
from zmq import EVENTS, POLLIN, POLLOUT
from zmq._typing import Literal
def unwrap_result(f):
    if future.done():
        return
    if poll_future.cancelled():
        try:
            future.cancel()
        except RuntimeError:
            pass
        return
    if f.exception():
        future.set_exception(poll_future.exception())
    else:
        evts = dict(poll_future.result())
        future.set_result(evts.get(self, 0))