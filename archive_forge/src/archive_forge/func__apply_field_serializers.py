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
def _apply_field_serializers(self, schema: core_schema.CoreSchema, serializers: list[Decorator[FieldSerializerDecoratorInfo]], computed_field: bool=False) -> core_schema.CoreSchema:
    """Apply field serializers to a schema."""
    if serializers:
        schema = copy(schema)
        if schema['type'] == 'definitions':
            inner_schema = schema['schema']
            schema['schema'] = self._apply_field_serializers(inner_schema, serializers)
            return schema
        else:
            ref = typing.cast('str|None', schema.get('ref', None))
            if ref is not None:
                schema = core_schema.definition_reference_schema(ref)
        serializer = serializers[-1]
        is_field_serializer, info_arg = inspect_field_serializer(serializer.func, serializer.info.mode, computed_field=computed_field)
        try:
            return_type = _decorators.get_function_return_type(serializer.func, serializer.info.return_type, self._types_namespace)
        except NameError as e:
            raise PydanticUndefinedAnnotation.from_name_error(e) from e
        if return_type is PydanticUndefined:
            return_schema = None
        else:
            return_schema = self.generate_schema(return_type)
        if serializer.info.mode == 'wrap':
            schema['serialization'] = core_schema.wrap_serializer_function_ser_schema(serializer.func, is_field_serializer=is_field_serializer, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
        else:
            assert serializer.info.mode == 'plain'
            schema['serialization'] = core_schema.plain_serializer_function_ser_schema(serializer.func, is_field_serializer=is_field_serializer, info_arg=info_arg, return_schema=return_schema, when_used=serializer.info.when_used)
    return schema