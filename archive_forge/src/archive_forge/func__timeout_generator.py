import abc
import atexit
import contextlib
import logging
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings
from . import constants, exceptions, portalocker
def _timeout_generator(self, timeout: typing.Optional[float], check_interval: typing.Optional[float]) -> typing.Iterator[int]:
    f_timeout = coalesce(timeout, self.timeout, 0.0)
    f_check_interval = coalesce(check_interval, self.check_interval, 0.0)
    yield 0
    i = 0
    start_time = time.perf_counter()
    while start_time + f_timeout > time.perf_counter():
        i += 1
        yield i
        since_start_time = time.perf_counter() - start_time
        time.sleep(max(0.001, i * f_check_interval - since_start_time))