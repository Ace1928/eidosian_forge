from __future__ import annotations as _annotations
import collections
import collections.abc
import dataclasses
import decimal
import inspect
import os
import typing
from enum import Enum
from functools import partial
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any, Callable, Iterable, TypeVar
import typing_extensions
from pydantic_core import (
from typing_extensions import get_args, get_origin
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from pydantic.types import Strict
from ..config import ConfigDict
from ..json_schema import JsonSchemaValue, update_json_schema
from . import _known_annotated_metadata, _typing_extra, _validators
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._schema_generation_shared import GetCoreSchemaHandler, GetJsonSchemaHandler
@dataclasses.dataclass(**slots_true)
class InnerSchemaValidator:
    """Use a fixed CoreSchema, avoiding interference from outward annotations."""
    core_schema: CoreSchema
    js_schema: JsonSchemaValue | None = None
    js_core_schema: CoreSchema | None = None
    js_schema_update: JsonSchemaValue | None = None

    def __get_pydantic_json_schema__(self, _schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        if self.js_schema is not None:
            return self.js_schema
        js_schema = handler(self.js_core_schema or self.core_schema)
        if self.js_schema_update is not None:
            js_schema.update(self.js_schema_update)
        return js_schema

    def __get_pydantic_core_schema__(self, _source_type: Any, _handler: GetCoreSchemaHandler) -> CoreSchema:
        return self.core_schema