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
def tuple_schema(self, schema: core_schema.TupleSchema) -> JsonSchemaValue:
    """Generates a JSON schema that matches a tuple schema e.g. `Tuple[int,
        str, bool]` or `Tuple[int, ...]`.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
    json_schema: JsonSchemaValue = {'type': 'array'}
    if 'variadic_item_index' in schema:
        variadic_item_index = schema['variadic_item_index']
        if variadic_item_index > 0:
            json_schema['minItems'] = variadic_item_index
            json_schema['prefixItems'] = [self.generate_inner(item) for item in schema['items_schema'][:variadic_item_index]]
        if variadic_item_index + 1 == len(schema['items_schema']):
            json_schema['items'] = self.generate_inner(schema['items_schema'][variadic_item_index])
        else:
            json_schema['items'] = True
    else:
        prefixItems = [self.generate_inner(item) for item in schema['items_schema']]
        if prefixItems:
            json_schema['prefixItems'] = prefixItems
        json_schema['minItems'] = len(prefixItems)
        json_schema['maxItems'] = len(prefixItems)
    self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
    return json_schema