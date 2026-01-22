import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
class TimerRequest:
    """
    Data object representing a countdown timer acquisition and release
    that is used between the ``TimerClient`` and ``TimerServer``.
    A negative ``expiration_time`` should be interpreted as a "release"
    request.

    .. note:: the type of ``worker_id`` is implementation specific.
              It is whatever the TimerServer and TimerClient implementations
              have on to uniquely identify a worker.
    """
    __slots__ = ['worker_id', 'scope_id', 'expiration_time']

    def __init__(self, worker_id: Any, scope_id: str, expiration_time: float):
        self.worker_id = worker_id
        self.scope_id = scope_id
        self.expiration_time = expiration_time

    def __eq__(self, other):
        if isinstance(other, TimerRequest):
            return self.worker_id == other.worker_id and self.scope_id == other.scope_id and (self.expiration_time == other.expiration_time)
        return False