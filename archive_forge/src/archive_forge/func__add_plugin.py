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
def _add_plugin(self, plugin: CoveragePlugin, specialized: list[CoveragePlugin] | None) -> None:
    """Add a plugin object.

        `plugin` is a :class:`CoveragePlugin` instance to add.  `specialized`
        is a list to append the plugin to.

        """
    plugin_name = f'{self.current_module}.{plugin.__class__.__name__}'
    if self.debug and self.debug.should('plugin'):
        self.debug.write(f'Loaded plugin {self.current_module!r}: {plugin!r}')
        labelled = LabelledDebug(f'plugin {self.current_module!r}', self.debug)
        plugin = DebugPluginWrapper(plugin, labelled)
    plugin._coverage_plugin_name = plugin_name
    plugin._coverage_enabled = True
    self.order.append(plugin)
    self.names[plugin_name] = plugin
    if specialized is not None:
        specialized.append(plugin)