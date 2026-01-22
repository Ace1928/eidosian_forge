from __future__ import annotations
import functools
import inspect
import os
import os.path
import sys
import threading
import traceback
from dataclasses import dataclass
from types import CodeType, FrameType
from typing import (
from coverage.debug import short_filename, short_stack
from coverage.types import (
@dataclass
class CodeInfo:
    """The information we want about each code object."""
    tracing: bool
    file_data: TTraceFileData | None
    byte_to_line: dict[int, int] | None