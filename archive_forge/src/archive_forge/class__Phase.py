from __future__ import annotations
import enum
import os.path
import string
import typing as T
from .. import coredata
from .. import mlog
from ..mesonlib import (
from .compilers import Compiler
class _Phase(enum.Enum):
    COMPILER = 'compiler'
    LINKER = 'linker'