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
def _installation_trace(self, frame: FrameType, event: str, arg: Any) -> TTraceFn | None:
    """Called on new threads, installs the real tracer."""
    sys.settrace(None)
    fn: TTraceFn | None = self._start_tracer()
    if fn:
        fn = fn(frame, event, arg)
    return fn