from __future__ import annotations
import abc
import dataclasses
import enum
import os
import pickle
import sys
import typing as t
from .constants import (
from .io import (
from .completion import (
from .util import (
@dataclasses.dataclass(frozen=True)
class FallbackDetail:
    """Details about controller fallback behavior."""
    reason: FallbackReason
    message: str