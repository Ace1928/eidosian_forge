from __future__ import annotations
import collections.abc as c
import contextlib
import functools
import sys
import threading
import queue
import typing as t
def wait_for_result(self) -> t.Any:
    """Wait for thread to exit and return the result or raise an exception."""
    result, exception = self._result.get()
    if exception:
        raise exception[1].with_traceback(exception[2])
    self.result = result
    return result