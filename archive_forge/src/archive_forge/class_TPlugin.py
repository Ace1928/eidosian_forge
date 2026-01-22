from __future__ import annotations
import os
import pathlib
from types import FrameType, ModuleType
from typing import (
class TPlugin(Protocol):
    """What all plugins have in common."""
    _coverage_plugin_name: str
    _coverage_enabled: bool