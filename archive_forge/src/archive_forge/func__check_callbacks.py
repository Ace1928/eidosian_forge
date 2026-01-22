import contextlib
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import (
@classmethod
def _check_callbacks(cls, oldvalue: Any) -> None:
    """
        Execute all needed callbacks if config value was changed.

        Parameters
        ----------
        oldvalue : Any
            Previous (old) config value.
        """
    if oldvalue == cls.get():
        return
    for callback in cls._subs:
        callback(cls)
    for callback in cls._once.pop(cls.get(), ()):
        callback(cls)