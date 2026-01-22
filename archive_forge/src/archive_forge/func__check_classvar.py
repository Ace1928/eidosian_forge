from __future__ import annotations as _annotations
import dataclasses
import sys
import types
import typing
from collections.abc import Callable
from functools import partial
from types import GetSetDescriptorType
from typing import TYPE_CHECKING, Any, Final
from typing_extensions import Annotated, Literal, TypeAliasType, TypeGuard, get_args, get_origin
def _check_classvar(v: type[Any] | None) -> bool:
    if v is None:
        return False
    return v.__class__ == typing.ClassVar.__class__ and getattr(v, '_name', None) == 'ClassVar'