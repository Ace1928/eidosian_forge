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
def _tuple_schema(self, tuple_type: Any) -> core_schema.CoreSchema:
    """Generate schema for a Tuple, e.g. `tuple[int, str]` or `tuple[int, ...]`."""
    typevars_map = get_standard_typevars_map(tuple_type)
    params = self._get_args_resolving_forward_refs(tuple_type)
    if typevars_map and params:
        params = tuple((replace_types(param, typevars_map) for param in params))
    if not params:
        if tuple_type in TUPLE_TYPES:
            return core_schema.tuple_schema([core_schema.any_schema()], variadic_item_index=0)
        else:
            return core_schema.tuple_schema([])
    elif params[-1] is Ellipsis:
        if len(params) == 2:
            return core_schema.tuple_schema([self.generate_schema(params[0])], variadic_item_index=0)
        else:
            raise ValueError('Variable tuples can only have one type')
    elif len(params) == 1 and params[0] == ():
        return core_schema.tuple_schema([])
    else:
        return core_schema.tuple_schema([self.generate_schema(param) for param in params])