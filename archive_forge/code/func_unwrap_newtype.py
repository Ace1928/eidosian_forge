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
def unwrap_newtype(typ: TypeOrCallableOrNone) -> Tuple[TypeOrCallableOrNone, Optional[str]]:
    return_name = None
    while hasattr(typ, '__name__') and hasattr(typ, '__supertype__'):
        if return_name is None:
            return_name = getattr(typ, '__name__')
        typ = getattr(typ, '__supertype__')
    return (typ, return_name)