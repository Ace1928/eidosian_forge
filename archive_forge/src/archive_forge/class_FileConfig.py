from __future__ import annotations
import logging # isort:skip
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from ..core.types import PathLike
from ..document import Document
from ..resources import Resources, ResourcesMode
@dataclass(frozen=True)
class FileConfig:
    filename: PathLike
    resources: Resources
    title: str