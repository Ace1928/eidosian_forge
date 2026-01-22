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
class PytestExperimentalApiWarning(PytestWarning, FutureWarning):
    """Warning category used to denote experiments in pytest.

    Use sparingly as the API might change or even be removed completely in a
    future version.
    """
    __module__ = 'pytest'

    @classmethod
    def simple(cls, apiname: str) -> 'PytestExperimentalApiWarning':
        return cls(f'{apiname} is an experimental api that may change over time')