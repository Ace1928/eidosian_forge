import abc
import logging
import threading
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.train.constants import _DEPRECATED_VALUE
from ray.util import log_once
from ray.util.annotations import PublicAPI
from ray.widgets import Template
class _BackgroundProcess:

    def __init__(self, fn: Callable):
        self._fn = fn
        self._process = None
        self._result = {}
        self._start_time = float('-inf')

    @property
    def is_running(self):
        return self._process and self._process.is_alive()

    @property
    def start_time(self):
        return self._start_time

    def start(self, *args, **kwargs):
        if self.is_running:
            return False
        self._result = {}

        def entrypoint():
            try:
                result = self._fn(*args, **kwargs)
            except Exception as e:
                self._result['exception'] = e
                return
            self._result['result'] = result
        self._process = threading.Thread(target=entrypoint)
        self._process.daemon = True
        self._process.start()
        self._start_time = time.time()

    def wait(self, timeout: Optional[float]=None) -> Any:
        """Waits for the background process to finish running. Waits until the
        background process has run for at least `timeout` seconds, counting from
        the time when the process was started."""
        if not self._process:
            return None
        time_remaining = None
        if timeout:
            elapsed = time.time() - self.start_time
            time_remaining = max(timeout - elapsed, 0)
        self._process.join(timeout=time_remaining)
        if self._process.is_alive():
            self._process = None
            raise TimeoutError(f'{getattr(self._fn, '__name__', str(self._fn))} did not finish running within the timeout of {timeout} seconds.')
        self._process = None
        exception = self._result.get('exception')
        if exception:
            raise exception
        result = self._result.get('result')
        self._result = {}
        return result