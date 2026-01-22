from __future__ import annotations
import os
import os.path
import sys
from types import FrameType
from typing import Any, Iterable, Iterator
from coverage.exceptions import PluginError
from coverage.misc import isolate_module
from coverage.plugin import CoveragePlugin, FileTracer, FileReporter
from coverage.types import (
def add_noop(self, plugin: CoveragePlugin) -> None:
    """Add a plugin that does nothing.

        This is only useful for testing the plugin support.

        """
    self._add_plugin(plugin, None)