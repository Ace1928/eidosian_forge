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
def _ensure_dataclass_instance_used_as_default_is_frozen(field: dataclasses.Field, default_instance: Any) -> None:
    """Ensure that a dataclass type used directly as a default value is marked as
    frozen."""
    assert dataclasses.is_dataclass(default_instance)
    cls = type(default_instance)
    if not cls.__dataclass_params__.frozen:
        warnings.warn(f'Mutable type {cls} is used as a default value for `{field.name}`. This is dangerous! Consider using `dataclasses.field(default_factory=...)` or marking {cls} as frozen.')