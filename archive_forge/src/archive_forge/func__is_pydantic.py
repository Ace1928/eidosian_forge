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
def _is_pydantic(cls: TypeForm[Any]) -> bool:
    if pydantic is not None and issubclass(cls, pydantic.BaseModel):
        return True
    if pydantic_v1 is not None and issubclass(cls, pydantic_v1.BaseModel):
        return True
    return False