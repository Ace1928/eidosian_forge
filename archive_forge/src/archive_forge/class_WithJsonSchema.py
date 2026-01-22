from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import (
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import (
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticUserError
@dataclasses.dataclass(**_internal_dataclass.slots_true)
class WithJsonSchema:
    """Usage docs: https://docs.pydantic.dev/2.6/concepts/json_schema/#withjsonschema-annotation

    Add this as an annotation on a field to override the (base) JSON schema that would be generated for that field.
    This provides a way to set a JSON schema for types that would otherwise raise errors when producing a JSON schema,
    such as Callable, or types that have an is-instance core schema, without needing to go so far as creating a
    custom subclass of pydantic.json_schema.GenerateJsonSchema.
    Note that any _modifications_ to the schema that would normally be made (such as setting the title for model fields)
    will still be performed.

    If `mode` is set this will only apply to that schema generation mode, allowing you
    to set different json schemas for validation and serialization.
    """
    json_schema: JsonSchemaValue | None
    mode: Literal['validation', 'serialization'] | None = None

    def __get_pydantic_json_schema__(self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        if mode != handler.mode:
            return handler(core_schema)
        if self.json_schema is None:
            raise PydanticOmit
        else:
            return self.json_schema

    def __hash__(self) -> int:
        return hash(type(self.mode))