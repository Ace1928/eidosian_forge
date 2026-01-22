from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def _build_schema_type_to_method(self) -> dict[core_schema.CoreSchemaType, Recurse]:
    mapping: dict[core_schema.CoreSchemaType, Recurse] = {}
    key: core_schema.CoreSchemaType
    for key in get_args(core_schema.CoreSchemaType):
        method_name = f'handle_{key.replace('-', '_')}_schema'
        mapping[key] = getattr(self, method_name, self._handle_other_schemas)
    return mapping