import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType
@enum.unique
class RuntimeTypeCheckState(enum.Enum):
    """Runtime type check state."""
    DISABLED = enum.auto()
    WARNINGS = enum.auto()
    ERRORS = enum.auto()