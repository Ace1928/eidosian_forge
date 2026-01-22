from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
@dataclasses.dataclass(frozen=True)
class PayloadConfig:
    """Configuration required to build a source tree payload for delegation."""
    files: list[tuple[str, str]]
    permissions: dict[str, int]