from __future__ import annotations
import functools
import os
import sys
from types import FrameType
from typing import (
from coverage import env
from coverage.config import CoverageConfig
from coverage.data import CoverageData
from coverage.debug import short_stack
from coverage.disposition import FileDisposition
from coverage.exceptions import ConfigError
from coverage.misc import human_sorted_items, isolate_module
from coverage.plugin import CoveragePlugin
from coverage.pytracer import PyTracer
from coverage.sysmon import SysMonitor
from coverage.types import (
def tracer_name(self) -> str:
    """Return the class name of the tracer we're using."""
    return self._trace_class.__name__