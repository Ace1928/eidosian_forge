from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TFileDisposition(Protocol):
    """A simple value type for recording what to do with a file."""
    original_filename: str
    canonical_filename: str
    source_filename: str | None
    trace: bool
    reason: str
    file_tracer: FileTracer | None
    has_dynamic_filename: bool