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
@dataclasses.dataclass(frozen=True)
class UnsupportedNestedTypeMessage:
    """Reason why a callable cannot be treated as a nested type."""
    message: str