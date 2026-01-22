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
def _add_send_event(self, kind, msg=None, kwargs=None, future=None):
    """Add a send event, returning the corresponding Future"""
    f = future or self._Future()
    if kind in ('send', 'send_multipart') and (not self._send_futures):
        flags = kwargs.get('flags', 0)
        nowait_kwargs = kwargs.copy()
        nowait_kwargs['flags'] = flags | _zmq.DONTWAIT
        send = getattr(self._shadow_sock, kind)
        finish_early = True
        try:
            r = send(msg, **nowait_kwargs)
        except _zmq.Again as e:
            if flags & _zmq.DONTWAIT:
                f.set_exception(e)
            else:
                finish_early = False
        except Exception as e:
            f.set_exception(e)
        else:
            f.set_result(r)
        if finish_early:
            if self._recv_futures:
                self._schedule_remaining_events()
            return f
    timer = _NoTimer
    if hasattr(_zmq, 'SNDTIMEO'):
        timeout_ms = self._shadow_sock.get(_zmq.SNDTIMEO)
        if timeout_ms >= 0:
            timer = self._add_timeout(f, timeout_ms * 0.001)
    _future_event = _FutureEvent(f, kind, kwargs=kwargs, msg=msg, timer=timer)
    self._send_futures.append(_future_event)
    f.add_done_callback(partial(self._remove_finished_future, event_list=self._send_futures, event=_future_event))
    self._add_io_state(POLLOUT)
    return f