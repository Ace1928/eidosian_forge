import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
@abc.abstractmethod
def _reap_worker(self, worker_id: Any) -> bool:
    """
        Reaps the given worker. Returns True if the worker has been
        successfully reaped, False otherwise. If any uncaught exception
        is thrown from this method, the worker is considered reaped
        and all associated timers will be removed.
        """