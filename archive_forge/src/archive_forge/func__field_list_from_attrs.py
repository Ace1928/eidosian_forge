from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
def _field_list_from_attrs(cls: TypeForm[Any], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    assert attr is not None
    attr.resolve_types(cls)
    field_list = []
    for attr_field in attr.fields(cls):
        if not attr_field.init:
            continue
        name = attr_field.name
        default = attr_field.default
        is_default_from_default_instance = False
        if default_instance not in MISSING_SINGLETONS:
            if hasattr(default_instance, name):
                default = getattr(default_instance, name)
                is_default_from_default_instance = True
            else:
                warnings.warn(f'Could not find field {name} in default instance {default_instance}, which has type {type(default_instance)},', stacklevel=2)
        elif default is attr.NOTHING:
            default = MISSING_NONPROP
        elif isinstance(default, attr.Factory):
            default = default.factory()
        assert attr_field.type is not None
        field_list.append(FieldDefinition.make(name=name, type_or_callable=attr_field.type, default=default, is_default_from_default_instance=is_default_from_default_instance, helptext=_docstrings.get_field_docstring(cls, name)))
    return field_list