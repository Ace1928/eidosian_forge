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
def union_schema(self, schema: core_schema.UnionSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a schema that allows values matching any of the given schemas.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    generated: list[JsonSchemaValue] = []
    choices = schema['choices']
    for choice in choices:
        choice_schema = choice[0] if isinstance(choice, tuple) else choice
        try:
            generated.append(self.generate_inner(choice_schema))
        except PydanticOmit:
            continue
        except PydanticInvalidForJsonSchema as exc:
            self.emit_warning('skipped-choice', exc.message)
    if len(generated) == 1:
        return generated[0]
    return self.get_flattened_anyof(generated)