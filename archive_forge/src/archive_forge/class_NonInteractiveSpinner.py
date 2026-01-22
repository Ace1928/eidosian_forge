import contextlib
import itertools
import logging
import sys
import time
from typing import IO, Generator, Optional
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation
class NonInteractiveSpinner(SpinnerInterface):

    def __init__(self, message: str, min_update_interval_seconds: float=60.0) -> None:
        self._message = message
        self._finished = False
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._update('started')

    def _update(self, status: str) -> None:
        assert not self._finished
        self._rate_limiter.reset()
        logger.info('%s: %s', self._message, status)

    def spin(self) -> None:
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._update('still running...')

    def finish(self, final_status: str) -> None:
        if self._finished:
            return
        self._update(f"finished with status '{final_status}'")
        self._finished = True