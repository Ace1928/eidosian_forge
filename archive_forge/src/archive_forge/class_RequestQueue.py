import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
class RequestQueue(abc.ABC):
    """
    Consumer queue holding timer acquisition/release requests
    """

    @abc.abstractmethod
    def size(self) -> int:
        """
        Returns the size of the queue at the time this method is called.
        Note that by the time ``get`` is called the size of the queue
        may have increased. The size of the queue should not decrease
        until the ``get`` method is called. That is, the following assertion
        should hold:

        size = q.size()
        res = q.get(size, timeout=0)
        assert size == len(res)

        -- or --

        size = q.size()
        res = q.get(size * 2, timeout=1)
        assert size <= len(res) <= size * 2
        """
        pass

    @abc.abstractmethod
    def get(self, size: int, timeout: float) -> List[TimerRequest]:
        """
        Gets up to ``size`` number of timer requests in a blocking fashion
        (no more than ``timeout`` seconds).
        """
        pass