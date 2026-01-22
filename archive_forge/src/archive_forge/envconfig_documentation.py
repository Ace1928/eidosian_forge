from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
Lookup binary in cross/native file and fallback to environment.

        Returns command with args as list if found, Returns `None` if nothing is
        found.
        