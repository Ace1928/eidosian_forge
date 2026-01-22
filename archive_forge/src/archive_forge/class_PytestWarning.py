import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
class PytestWarning(UserWarning):
    """Base class for all warnings emitted by pytest."""
    __module__ = 'pytest'