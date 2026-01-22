import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
@final
class PytestConfigWarning(PytestWarning):
    """Warning emitted for configuration issues."""
    __module__ = 'pytest'