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
def _field_list_from_typeddict(cls: TypeForm[Any], default_instance: DefaultInstance) -> Union[List[FieldDefinition], UnsupportedNestedTypeMessage]:
    field_list = []
    valid_default_instance = default_instance not in MISSING_SINGLETONS and default_instance is not EXCLUDE_FROM_CALL
    assert not valid_default_instance or isinstance(default_instance, dict)
    total = getattr(cls, '__total__', True)
    assert isinstance(total, bool)
    assert not valid_default_instance or isinstance(default_instance, dict)
    for name, typ in _resolver.get_type_hints(cls, include_extras=True).items():
        typ_origin = get_origin(typ)
        is_default_from_default_instance = False
        if valid_default_instance and name in cast(dict, default_instance):
            default = cast(dict, default_instance)[name]
            is_default_from_default_instance = True
        elif typ_origin is Required and total is False:
            default = MISSING_PROP
        elif total is False:
            default = EXCLUDE_FROM_CALL
            if is_nested_type(typ, MISSING_NONPROP):
                pass
        elif typ_origin is NotRequired:
            default = EXCLUDE_FROM_CALL
        else:
            default = MISSING_PROP
        if default is EXCLUDE_FROM_CALL and is_nested_type(typ, MISSING_NONPROP):
            default = MISSING_NONPROP
        if typ_origin in (Required, NotRequired):
            args = get_args(typ)
            assert len(args) == 1, 'typing.Required[] and typing.NotRequired[T] require a concrete type T.'
            typ = args[0]
            del args
        field_list.append(FieldDefinition.make(name=name, type_or_callable=typ, default=default, is_default_from_default_instance=is_default_from_default_instance, helptext=_docstrings.get_field_docstring(cls, name)))
    return field_list