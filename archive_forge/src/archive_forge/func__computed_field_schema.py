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
def _computed_field_schema(self, d: Decorator[ComputedFieldInfo], field_serializers: dict[str, Decorator[FieldSerializerDecoratorInfo]]) -> core_schema.ComputedField:
    try:
        return_type = _decorators.get_function_return_type(d.func, d.info.return_type, self._types_namespace)
    except NameError as e:
        raise PydanticUndefinedAnnotation.from_name_error(e) from e
    if return_type is PydanticUndefined:
        raise PydanticUserError('Computed field is missing return type annotation or specifying `return_type` to the `@computed_field` decorator (e.g. `@computed_field(return_type=int|str)`)', code='model-field-missing-annotation')
    return_type = replace_types(return_type, self._typevars_map)
    d.info = dataclasses.replace(d.info, return_type=return_type)
    return_type_schema = self.generate_schema(return_type)
    return_type_schema = self._apply_field_serializers(return_type_schema, filter_field_decorator_info_by_field(field_serializers.values(), d.cls_var_name), computed_field=True)
    alias_generator = self._config_wrapper.alias_generator
    if alias_generator is not None:
        self._apply_alias_generator_to_computed_field_info(alias_generator=alias_generator, computed_field_info=d.info, computed_field_name=d.cls_var_name)

    def set_computed_field_metadata(schema: CoreSchemaOrField, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema['readOnly'] = True
        title = d.info.title
        if title is not None:
            json_schema['title'] = title
        description = d.info.description
        if description is not None:
            json_schema['description'] = description
        examples = d.info.examples
        if examples is not None:
            json_schema['examples'] = to_jsonable_python(examples)
        json_schema_extra = d.info.json_schema_extra
        if json_schema_extra is not None:
            add_json_schema_extra(json_schema, json_schema_extra)
        return json_schema
    metadata = build_metadata_dict(js_annotation_functions=[set_computed_field_metadata])
    return core_schema.computed_field(d.cls_var_name, return_schema=return_type_schema, alias=d.info.alias, metadata=metadata)