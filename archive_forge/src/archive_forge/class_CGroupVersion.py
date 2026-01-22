from __future__ import annotations
import abc
import dataclasses
import enum
import os
import typing as t
from .constants import (
from .util import (
from .data import (
from .become import (
class CGroupVersion(enum.Enum):
    """The control group version(s) required by a container."""
    NONE = 'none'
    V1_ONLY = 'v1-only'
    V2_ONLY = 'v2-only'
    V1_V2 = 'v1-v2'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}.{self.name}'