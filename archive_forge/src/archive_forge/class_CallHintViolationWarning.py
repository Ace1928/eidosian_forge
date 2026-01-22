import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType
class CallHintViolationWarning(UserWarning):
    """Warning raised when a type hint is violated during a function call."""
    pass