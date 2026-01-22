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
@dataclasses.dataclass(frozen=True)
class CompletionConfig(metaclass=abc.ABCMeta):
    """Base class for completion configuration."""
    name: str

    @property
    @abc.abstractmethod
    def is_default(self) -> bool:
        """True if the completion entry is only used for defaults, otherwise False."""