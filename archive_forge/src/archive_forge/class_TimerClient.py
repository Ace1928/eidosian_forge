import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
class TimerClient(abc.ABC):
    """
    Client library to acquire and release countdown timers by communicating
    with the TimerServer.
    """

    @abc.abstractmethod
    def acquire(self, scope_id: str, expiration_time: float) -> None:
        """
        Acquires a timer for the worker that holds this client object
        given the scope_id and expiration_time. Typically registers
        the timer with the TimerServer.
        """
        pass

    @abc.abstractmethod
    def release(self, scope_id: str):
        """
        Releases the timer for the ``scope_id`` on the worker this
        client represents. After this method is
        called, the countdown timer on the scope is no longer in effect.
        """
        pass