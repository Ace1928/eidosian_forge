from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
@dataclass(**slots_true)
class ValidatorDecoratorInfo:
    """A container for data from `@validator` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@validator'.
        fields: A tuple of field names the validator should be called on.
        mode: The proposed validator mode.
        each_item: For complex objects (sets, lists etc.) whether to validate individual
            elements rather than the whole object.
        always: Whether this method and other validators should be called even if the value is missing.
        check_fields: Whether to check that the fields actually exist on the model.
    """
    decorator_repr: ClassVar[str] = '@validator'
    fields: tuple[str, ...]
    mode: Literal['before', 'after']
    each_item: bool
    always: bool
    check_fields: bool | None