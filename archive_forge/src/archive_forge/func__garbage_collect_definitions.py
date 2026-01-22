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
def _garbage_collect_definitions(self, schema: JsonSchemaValue) -> None:
    visited_defs_refs: set[DefsRef] = set()
    unvisited_json_refs = _get_all_json_refs(schema)
    while unvisited_json_refs:
        next_json_ref = unvisited_json_refs.pop()
        next_defs_ref = self.json_to_defs_refs[next_json_ref]
        if next_defs_ref in visited_defs_refs:
            continue
        visited_defs_refs.add(next_defs_ref)
        unvisited_json_refs.update(_get_all_json_refs(self.definitions[next_defs_ref]))
    self.definitions = {k: v for k, v in self.definitions.items() if k in visited_defs_refs}