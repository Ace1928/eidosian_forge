from __future__ import annotations as _annotations
import dataclasses
import sys
from functools import partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, cast, overload
from pydantic_core import core_schema
from pydantic_core import core_schema as _core_schema
from typing_extensions import Annotated, Literal, TypeAlias
from . import GetCoreSchemaHandler as _GetCoreSchemaHandler
from ._internal import _core_metadata, _decorators, _generics, _internal_dataclass
from .annotated_handlers import GetCoreSchemaHandler
from .errors import PydanticUserError
class ModelWrapValidator(Protocol[_ModelType]):
    """A @model_validator decorated function signature. This is used when `mode='wrap'`."""

    def __call__(self, cls: type[_ModelType], value: Any, handler: ModelWrapValidatorHandler[_ModelType], info: _core_schema.ValidationInfo, /) -> _ModelType:
        ...