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
def _apply_single_annotation(self, schema: core_schema.CoreSchema, metadata: Any) -> core_schema.CoreSchema:
    from ..fields import FieldInfo
    if isinstance(metadata, FieldInfo):
        for field_metadata in metadata.metadata:
            schema = self._apply_single_annotation(schema, field_metadata)
        if metadata.discriminator is not None:
            schema = self._apply_discriminator_to_union(schema, metadata.discriminator)
        return schema
    if schema['type'] == 'nullable':
        inner = schema.get('schema', core_schema.any_schema())
        inner = self._apply_single_annotation(inner, metadata)
        if inner:
            schema['schema'] = inner
        return schema
    original_schema = schema
    ref = schema.get('ref', None)
    if ref is not None:
        schema = schema.copy()
        new_ref = ref + f'_{repr(metadata)}'
        if new_ref in self.defs.definitions:
            return self.defs.definitions[new_ref]
        schema['ref'] = new_ref
    elif schema['type'] == 'definition-ref':
        ref = schema['schema_ref']
        if ref in self.defs.definitions:
            schema = self.defs.definitions[ref].copy()
            new_ref = ref + f'_{repr(metadata)}'
            if new_ref in self.defs.definitions:
                return self.defs.definitions[new_ref]
            schema['ref'] = new_ref
    maybe_updated_schema = _known_annotated_metadata.apply_known_metadata(metadata, schema.copy())
    if maybe_updated_schema is not None:
        return maybe_updated_schema
    return original_schema