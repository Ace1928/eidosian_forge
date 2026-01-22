from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import inspect
import re
import sys
import typing
import warnings
from contextlib import contextmanager
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from itertools import chain
from operator import attrgetter
from types import FunctionType, LambdaType, MethodType
from typing import (
from warnings import warn
from pydantic_core import CoreSchema, PydanticUndefined, core_schema, to_jsonable_python
from typing_extensions import Annotated, Literal, TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasGenerator
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _core_utils, _decorators, _discriminated_union, _known_annotated_metadata, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadataHandler, build_metadata_dict
from ._core_utils import (
from ._decorators import (
from ._fields import collect_dataclass_fields, get_type_hints_infer_globalns
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, has_instance_in_type, recursively_defined_type_refs, replace_types
from ._schema_generation_shared import (
from ._typing_extra import is_finalvar
from ._utils import lenient_issubclass
def _namedtuple_schema(self, namedtuple_cls: Any, origin: Any) -> core_schema.CoreSchema:
    """Generate schema for a NamedTuple."""
    with self.defs.get_schema_or_ref(namedtuple_cls) as (namedtuple_ref, maybe_schema):
        if maybe_schema is not None:
            return maybe_schema
        typevars_map = get_standard_typevars_map(namedtuple_cls)
        if origin is not None:
            namedtuple_cls = origin
        annotations: dict[str, Any] = get_type_hints_infer_globalns(namedtuple_cls, include_extras=True, localns=self._types_namespace)
        if not annotations:
            annotations = {k: Any for k in namedtuple_cls._fields}
        if typevars_map:
            annotations = {field_name: replace_types(annotation, typevars_map) for field_name, annotation in annotations.items()}
        arguments_schema = core_schema.arguments_schema([self._generate_parameter_schema(field_name, annotation, default=namedtuple_cls._field_defaults.get(field_name, Parameter.empty)) for field_name, annotation in annotations.items()], metadata=build_metadata_dict(js_prefer_positional_arguments=True))
        return core_schema.call_schema(arguments_schema, namedtuple_cls, ref=namedtuple_ref)