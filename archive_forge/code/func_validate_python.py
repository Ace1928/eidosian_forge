from __future__ import annotations as _annotations
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Set, TypeVar, Union, cast, final, overload
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator, Some
from typing_extensions import Literal, get_args, is_typeddict
from pydantic.errors import PydanticUserError
from pydantic.main import BaseModel
from ._internal import _config, _generate_schema, _typing_extra
from .config import ConfigDict
from .json_schema import (
from .plugin._schema_validator import create_schema_validator
def validate_python(self, __object: Any, *, strict: bool | None=None, from_attributes: bool | None=None, context: dict[str, Any] | None=None) -> T:
    """Validate a Python object against the model.

        Args:
            __object: The Python object to validate against the model.
            strict: Whether to strictly check types.
            from_attributes: Whether to extract data from object attributes.
            context: Additional context to pass to the validator.

        !!! note
            When using `TypeAdapter` with a Pydantic `dataclass`, the use of the `from_attributes`
            argument is not supported.

        Returns:
            The validated object.
        """
    return self.validator.validate_python(__object, strict=strict, from_attributes=from_attributes, context=context)