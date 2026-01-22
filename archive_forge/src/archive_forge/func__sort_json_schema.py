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
def _sort_json_schema(value: JsonSchemaValue, parent_key: str | None=None) -> JsonSchemaValue:
    if isinstance(value, dict):
        sorted_dict: dict[str, JsonSchemaValue] = {}
        keys = value.keys()
        if parent_key != 'properties' and parent_key != 'default':
            keys = sorted(keys)
        for key in keys:
            sorted_dict[key] = _sort_json_schema(value[key], parent_key=key)
        return sorted_dict
    elif isinstance(value, list):
        sorted_list: list[JsonSchemaValue] = []
        for item in value:
            sorted_list.append(_sort_json_schema(item, parent_key))
        return sorted_list
    else:
        return value