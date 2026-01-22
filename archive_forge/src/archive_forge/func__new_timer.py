from timeit import default_timer
from types import TracebackType
from typing import (
from .decorator import decorate
def _new_timer(self):
    return self.__class__(self._metric, self._callback_name)