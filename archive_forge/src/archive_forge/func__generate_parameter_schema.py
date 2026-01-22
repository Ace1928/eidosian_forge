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
def _generate_parameter_schema(self, name: str, annotation: type[Any], default: Any=Parameter.empty, mode: Literal['positional_only', 'positional_or_keyword', 'keyword_only'] | None=None) -> core_schema.ArgumentsParameter:
    """Prepare a ArgumentsParameter to represent a field in a namedtuple or function signature."""
    from ..fields import FieldInfo
    if default is Parameter.empty:
        field = FieldInfo.from_annotation(annotation)
    else:
        field = FieldInfo.from_annotated_attribute(annotation, default)
    assert field.annotation is not None, 'field.annotation should not be None when generating a schema'
    source_type, annotations = (field.annotation, field.metadata)
    with self.field_name_stack.push(name):
        schema = self._apply_annotations(source_type, annotations)
    if not field.is_required():
        schema = wrap_default(field, schema)
    parameter_schema = core_schema.arguments_parameter(name, schema)
    if mode is not None:
        parameter_schema['mode'] = mode
    if field.alias is not None:
        parameter_schema['alias'] = field.alias
    else:
        alias_generator = self._config_wrapper.alias_generator
        if isinstance(alias_generator, AliasGenerator) and alias_generator.alias is not None:
            parameter_schema['alias'] = alias_generator.alias(name)
        elif isinstance(alias_generator, Callable):
            parameter_schema['alias'] = alias_generator(name)
    return parameter_schema