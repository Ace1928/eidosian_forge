import collections.abc
import copy
import dataclasses
import inspect
import sys
import types
import warnings
from typing import (
from typing_extensions import Annotated, Self, get_args, get_origin, get_type_hints
from . import _fields, _unsafe_cache
from ._typing import TypeForm
@_unsafe_cache.unsafe_cache(maxsize=1024)
def unwrap_newtype_and_narrow_subtypes(typ: TypeOrCallable, default_instance: Any) -> TypeOrCallable:
    """Type narrowing: if we annotate as Animal but specify a default instance of Cat,
    we should parse as Cat.

    This should generally only be applied to fields used as nested structures, not
    individual arguments/fields. (if a field is annotated as Union[int, str], and a
    string default is passed in, we don't want to narrow the type to always be
    strings!)"""
    typ, unused_name = unwrap_newtype(typ)
    del unused_name
    try:
        potential_subclass = type(default_instance)
        if potential_subclass is type:
            return typ
        superclass = unwrap_annotated(typ)[0]
        if get_origin(superclass) is Union:
            return typ
        if superclass is Any or issubclass(potential_subclass, superclass):
            if get_origin(typ) is Annotated:
                return Annotated.__class_getitem__((potential_subclass,) + get_args(typ)[1:])
            typ = cast(TypeOrCallable, potential_subclass)
    except TypeError:
        pass
    return typ