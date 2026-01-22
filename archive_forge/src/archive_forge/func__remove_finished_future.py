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
@staticmethod
def _remove_finished_future(future, event_list, event=None):
    """Make sure that futures are removed from the event list when they resolve

        Avoids delaying cleanup until the next send/recv event,
        which may never come.
        """
    if not event_list:
        return
    try:
        event_list.remove(event)
    except ValueError:
        return