from __future__ import annotations
import dataclasses
import enum
import os
import sys
import typing as t
from .util import (
from .util_common import (
from .metadata import (
from .data import (
from .host_configs import (
@dataclasses.dataclass(frozen=True)
class ContentConfig:
    """Configuration for all content."""
    modules: ModulesConfig
    python_versions: tuple[str, ...]
    py2_support: bool