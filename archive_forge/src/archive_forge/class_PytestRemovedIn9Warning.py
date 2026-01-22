import dataclasses
import inspect
from types import FunctionType
from typing import Any
from typing import final
from typing import Generic
from typing import Type
from typing import TypeVar
import warnings
class PytestRemovedIn9Warning(PytestDeprecationWarning):
    """Warning class for features that will be removed in pytest 9."""
    __module__ = 'pytest'