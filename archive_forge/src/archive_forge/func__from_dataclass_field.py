from __future__ import annotations as _annotations
import dataclasses
import inspect
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, Unpack
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
@staticmethod
def _from_dataclass_field(dc_field: DataclassField[Any]) -> FieldInfo:
    """Return a new `FieldInfo` instance from a `dataclasses.Field` instance.

        Args:
            dc_field: The `dataclasses.Field` instance to convert.

        Returns:
            The corresponding `FieldInfo` instance.

        Raises:
            TypeError: If any of the `FieldInfo` kwargs does not match the `dataclass.Field` kwargs.
        """
    default = dc_field.default
    if default is dataclasses.MISSING:
        default = PydanticUndefined
    if dc_field.default_factory is dataclasses.MISSING:
        default_factory: typing.Callable[[], Any] | None = None
    else:
        default_factory = dc_field.default_factory
    dc_field_metadata = {k: v for k, v in dc_field.metadata.items() if k in _FIELD_ARG_NAMES}
    return Field(default=default, default_factory=default_factory, repr=dc_field.repr, **dc_field_metadata)