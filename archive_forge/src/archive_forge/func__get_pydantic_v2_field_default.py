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
def _get_pydantic_v2_field_default(name: str, field: pydantic.fields.FieldInfo, parent_default_instance: DefaultInstance) -> Tuple[Any, bool]:
    """Helper for getting the default instance for a Pydantic field."""
    if parent_default_instance not in MISSING_SINGLETONS and parent_default_instance is not None:
        if hasattr(parent_default_instance, name):
            return (getattr(parent_default_instance, name), True)
        else:
            warnings.warn(f'Could not find field {name} in default instance {parent_default_instance}, which has type {type(parent_default_instance)},', stacklevel=2)
    if not field.is_required():
        return (field.get_default(call_default_factory=True), False)
    return (MISSING_NONPROP, False)