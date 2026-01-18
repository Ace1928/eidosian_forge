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
@deprecated('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', category=None)
@final
def tuple_positional_schema(self, schema: core_schema.TupleSchema) -> JsonSchemaValue:
    """Replaced by `tuple_schema`."""
    warnings.warn('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', PydanticDeprecatedSince26, stacklevel=2)
    return self.tuple_schema(schema)