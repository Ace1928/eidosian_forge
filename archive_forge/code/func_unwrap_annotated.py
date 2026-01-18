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
def unwrap_annotated(typ: TypeOrCallable, search_type: TypeForm[MetadataType]=cast(TypeForm[Any], Any)) -> Tuple[TypeOrCallable, Tuple[MetadataType, ...]]:
    """Helper for parsing typing.Annotated types.

    Examples:
    - int, int => (int, ())
    - Annotated[int, 1], int => (int, (1,))
    - Annotated[int, "1"], int => (int, ())
    """
    targets = tuple((x for x in getattr(typ, '__tyro_markers__', tuple()) if search_type is Any or isinstance(x, search_type)))
    assert isinstance(targets, tuple)
    if not hasattr(typ, '__metadata__'):
        return (typ, targets)
    args = get_args(typ)
    assert len(args) >= 2
    targets += tuple((x for x in targets + args[1:] if search_type is Any or isinstance(x, search_type)))
    targets += tuple((x for x in getattr(args[0], '__tyro_markers__', tuple()) if search_type is Any or isinstance(x, search_type)))
    return (args[0], targets)